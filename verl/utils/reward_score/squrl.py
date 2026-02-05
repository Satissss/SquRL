import re
import os
from typing import Dict, Tuple, Optional, List, Union
from func_timeout import func_timeout, FunctionTimedOut
import ast
from urllib import request, error
from .exec_eval import eval_exec_match
import json
import random

# Default available actors for SQL generation
DEFAULT_AVAILABLE_ACTORS = [
    "RSLSQLBiDirParser", "MACSQLCoTParser", "CHESSSelectorParser", "LinkAlignParser",
    "MACSQLGenerator", "CHESSGenerator", "RSLSQLGenerator", "LinkAlignGenerator",
    "ChessScaler", "MACSQLScaler", "RSLSQLScaler",
    "LinkAlignOptimizer", "MACSQLOptimizer", "CHESSOptimizer", "RSLSQLOptimizer",
    "FastExecSelector", "CHESSSelector", "ChaseSelector"
]

# Scoring constants
FORMAT_REWARD = 0.5


def validate_response_str(response_str: str, can_actors: List[str]=None) -> Tuple[bool, List]:
    """从LLM rollout生成的response字符串中解析出正确的actor列表
    
    Args:
        response_str: LLM生成的response字符串
        can_actors: 可选的候选actor列表，如果为None则使用默认列表
        
    Returns:
        Tuple[bool, List]: (是否验证成功, actor列表)
            - 成功返回 (True, parsed_actor_list)
            - 失败返回 (False, [])
    """
    if can_actors is not None:
        valid_actor_list = can_actors
    else:
        valid_actor_list = DEFAULT_AVAILABLE_ACTORS
    
    try:
        # 步骤1: 提取 <answer>...</answer> 标签中的内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, response_str, re.DOTALL)
        
        if not answer_matches:
            print("[Error] No <answer> tags found in response")
            return False, []
        
        # 取最后一个匹配的answer标签内容
        answer_content = answer_matches[-1].strip()
        
        # 步骤2: 从answer内容中提取 ```list...``` 格式的列表
        answer_content = response_str
        list_pattern = r'```list\s*(.*?)\s*```'
        list_matches = re.findall(list_pattern, answer_content, re.DOTALL)
        
        if not list_matches:
            print("[Error] No ```list...``` format found in answer")
            return False, []
        
        # 取最后一个匹配的列表内容
        list_str = list_matches[-1].strip()
        
        # 步骤3: 使用ast.literal_eval安全地解析字符串为Python列表
        try:
            parsed_list = ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            print(f"[Error] Failed to parse list string: {e}")
            return False, []
        
        # 确保解析结果是列表
        if not isinstance(parsed_list, list):
            print("[Error] Parsed result is not a list")
            return False, []
        
        # 步骤4: 递归验证所有actor是否在valid_actor_list中
        def validate_actors(actor_list: Union[List, str]) -> bool:
            """递归验证actor列表（支持嵌套）"""
            if isinstance(actor_list, str):
                # 如果是字符串，检查是否在valid_actor_list中
                if actor_list not in valid_actor_list:
                    print(f"[Error] Invalid actor: {actor_list}")
                    return False
                return True
            elif isinstance(actor_list, list):
                # 如果是列表，递归验证每个元素
                for item in actor_list:
                    if not validate_actors(item):
                        return False
                return True
            else:
                # 不是字符串也不是列表，无效
                print(f"[Error] Invalid actor type: {type(actor_list)}")
                return False
        
        # 验证所有actor
        if not validate_actors(parsed_list):
            return False, []
        
        # 步骤5: 验证成功，返回解析的列表


        return True, parsed_list
        
    except Exception as e:
        print(f"[Error] Unexpected error during validation: {e}")
        return False, []

def http_post_json(url: str, payload: dict, timeout: float = 720):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = body
            return status, data
    except error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = body
        return e.code, data
    except Exception as e:
        return 0, str(e)

def compute_score_batch(batch_responses: dict, 
                        api_port: int = 6517,
                        api_timeout: int = 1500, 
                        batch_size: int = 2):
    """批量计算响应分数
    
    Args:
        batch_responses: 响应批次字典
        api_port: API服务器端口
        api_timeout: API请求超时时间（秒）
        batch_size: 批次大小
        
    Returns:
        分数结果字典
    """
    # 最终结果字典，存储每个index的分数
    result_scores = {}
    # 存储验证通过的数据，用于后续批量请求
    # 格式: {instance_id: [(index, task_lis), ...]}
    valid_data = {}
    
    # 第一步：验证所有response_str，并初始化分数
    for instance_id, response_list in batch_responses.items():
        valid_data[instance_id] = []
        
        for index, response_str, can_actors in response_list:
            # 调用验证函数
            is_valid, task_lis = validate_response_str(response_str, can_actors)
            
            if is_valid:
                # 验证通过，初始分数为 FORMAT_REWARD
                result_scores[index] = FORMAT_REWARD
                # 保存用于后续批量请求
                valid_data[instance_id].append((index, task_lis))
            else:
                # 验证失败，分数为负的FORMAT_REWARD
                result_scores[index] = -abs(FORMAT_REWARD)


    # 第二步：分批发送POST请求
    # 收集所有需要请求的instance_id
    instance_ids_to_process = [
        instance_id for instance_id, data in valid_data.items() if len(data) > 0
    ]
    
    # 分批处理
    for batch_start in range(0, len(instance_ids_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(instance_ids_to_process))
        current_batch_ids = instance_ids_to_process[batch_start:batch_end]
        
        # 构建当前批次的payload
        payload = {}
        for instance_id in current_batch_ids:
            # 提取该instance_id下所有验证通过的task_lis
            task_lis_list = [task_lis for _, task_lis in valid_data[instance_id]]
            payload[instance_id] = task_lis_list
        print(payload)
        
        # 发送POST请求
        try:
            url = f"http://127.0.0.1:{api_port}/api/run_batch"
            status, batch_scores = http_post_json(url, payload, timeout=api_timeout)
            print(batch_scores)
            
            # 还原每个index的分数
            for instance_id in current_batch_ids:
                score_list = batch_scores.get(instance_id, [])
                index_task_pairs = valid_data[instance_id]
                
                # 确保返回的分数列表长度与输入一致
                if len(score_list) == len(index_task_pairs):
                    for (index, _), score in zip(index_task_pairs, score_list):
                        # 将评测分数加到已有的FORMAT_REWARD上
                        result_scores[index] += score
                else:
                    print(f"[Warning] Score list length mismatch for instance_id: {instance_id}")
                    
        except Exception as e:
            print(f"[Error] Failed to process batch {batch_start}-{batch_end}: {str(e)}")
            # 请求失败时，这些index保持原有的FORMAT_REWARD分数
    
    return result_scores