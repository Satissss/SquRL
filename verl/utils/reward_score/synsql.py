import re
import os
import sys
sys.path.append('..')
from typing import Dict, Tuple, Optional, List, Union
from func_timeout import func_timeout, FunctionTimedOut
import requests
import ast
from urllib import request, error
from .exec_eval import eval_exec_match
import signal
import json
import random

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, processed_str, re.DOTALL))

    if not think_matches:
        print("[Error] No valid think tags found")
        final_think = None
    else:
        final_think = think_matches[-1].group(1).strip()
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, final_think, processed_str
        
    final_answer = matches[-1].group(1).strip()

    return final_answer, final_think, processed_str

def parse_sql_from_answer(answer_text: str) -> Optional[str]:
    """Parses SQL from the model's answer text.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        
    Returns:
        SQL string, or None if no SQL is found
    """
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))
    
    if not matches:
        print("[Error] No valid SQL tags found")
        return None
    
    print(f"[Parsed SQL]: {matches[-1].group(1).strip()}")
    return matches[-1].group(1).strip()

def validate_response_structure(answer_str: str, processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("Tag sequence validation passed")

    # Extract SQL from answer text
    if validation_passed:
        pred_sql = parse_sql_from_answer(answer_str)
        if not pred_sql:
            validation_passed = False
    else:
        pred_sql = None

    return pred_sql, validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
    Returns:
        Total score (sum of format and answer rewards)
    """
    FORMAT_REWARD = 1
    EXEC_REWARD = 2
    RESULT_REWARD = 3

    LIMIT_LENGTH = 2048

    total_score = 0
    print("\n" + "="*80)
    print(" Processing New NL2SQL Sample ".center(80, '='))

    # Parse ground truth data
    db_name = ground_truth.get('db_id', '').replace('\n', '').strip()
    gold_sql = re.sub(r'\s+', ' ', ground_truth.get('sql', ''))
    # Extract model answer
    answer_text, think_text, processed_str = extract_solution(solution_str)
    # print(f"\n[Model's Response] {processed_str}")

    # Format Reward
    pred_sql, format_correct = validate_response_structure(answer_text, processed_str)
    format_score = FORMAT_REWARD if format_correct else -abs(FORMAT_REWARD)
    print(f"\n[Format validation] {'PASS' if format_correct else 'FAIL'}")
    print(f"[Format score]: {format_score}")

    db_path = os.path.join('data/NL2SQL/SynSQL-2.5M/databases', db_name, db_name + '.sqlite')

    exec_score = 0
    result_score = 0
    if format_correct and pred_sql:
        # Validate Exec Score
        pred_sql = re.sub(r'\s+', ' ', pred_sql)
        print(f"[DB NAME]: {db_name}")
        print(f"[Gold SQL]: {gold_sql}")
        print(f"[Pred SQL]: {pred_sql}")
        exec_status = 'Unexecutable'

        try:
            exec_status = func_timeout(
                timeout=30,
                func=eval_exec_match,
                args=(db_path, pred_sql, gold_sql),
                kwargs={
                    'plug_value': False, 
                    'keep_distinct': False, 
                    'progress_bar_for_each_datapoint': False
                }
            )
        except FunctionTimedOut:
            exec_status = 'Unexecutable'
        print(f"[Exec status]: {exec_status}")

        if exec_status == 'Unexecutable':
            exec_score = -abs(EXEC_REWARD)
            result_score = 0
        elif exec_status == 'Gold Error':
            exec_score = 0
            result_score = 0        
        elif exec_status == 'Mismatch':
            exec_score = EXEC_REWARD
            result_score = -abs(RESULT_REWARD)
        elif exec_status == 'Match':
            exec_score = EXEC_REWARD
            result_score = RESULT_REWARD

    # Length Reward v1: 鼓励输出接近 LIMIT_LENGTH，同时增加 SQL 在 answer 中的比例，但是要在 SQL 可执行才有意义
    # Length Reward v2: 更严格的长度奖励，只有 match 才有分，且比例更小 1 分
    # if format_correct and (exec_status == 'Mismatch' or exec_status == 'Match'):
    if format_correct and exec_status == 'Match':
        pos_length = len(think_text) + len(answer_text)
        if pos_length <= LIMIT_LENGTH:
            sql_in_answer_sub_score = len(pred_sql) / len(answer_text)
            length_sub_score = pos_length / LIMIT_LENGTH * 0.5
            length_score = length_sub_score + sql_in_answer_sub_score
            print(f"[Length pos_length]: {pos_length}")
            print(f"[Length pred_sql]: {len(pred_sql)}")
            print(f"[Length answer_text]: {len(answer_text)}")
        else:
            sql_in_answer_sub_score = len(pred_sql) / len(answer_text)
            length_score = 0.5 + sql_in_answer_sub_score
    else:
        length_score = 0

    

    total_score = format_score + exec_score + result_score + length_score

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f" -- Format Score: {format_score}")
    print(f" -- Exec Score: {exec_score}")
    print(f" -- Result Score: {result_score}")
    print(f" -- Length Score: {length_score}")
    print(f" -- Total Score: {total_score}")
    print("="*80 + "\n")

    return total_score


def validate_response_str(response_str: str, can_actors: List[str]=None) -> Tuple[bool, List]:
    """从LLM rollout生成的response字符串中解析出正确的actor列表
    
    Args:
        response_str: LLM生成的response字符串
        
    Returns:
        Tuple[bool, List]: (是否验证成功, actor列表)
            - 成功返回 (True, parsed_actor_list)
            - 失败返回 (False, [])
    """
    # return True, ['MACSQLGenerator']
    valid_actor_list = ['RSLSQLBiDirParser', 'MACSQLCoTParser', 'CHESSSelectorParser', 
                        'LinkAlignParser', 'MACSQLGenerator', 'CHESSGenerator', 
                        'RSLSQLGenerator', 'LinkAlignGenerator', 'ChessScaler', 
                        'MACSQLScaler', 'RSLSQLScaler', 'LinkAlignOptimizer', 
                        'MACSQLOptimizer', 'CHESSOptimizer', 'RSLSQLOptimizer', 
                        'FastExecSelector', 'CHESSSelector']
    
    if can_actors is not None:
        valid_actor_list = can_actors
    
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

def compute_actor_score(parsed_list, can_actors) -> float:
    def flatten_list(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten_list(item))
            else:
                result.append(item)
        return result
    
    def has_nested_list(lst):
        for item in lst:
            if isinstance(item, list) and len(item) > 1:
                return True
        return False
    
    # 展平parsed_list获取所有actor
    flattened_actors = flatten_list(parsed_list)
    
    # 1. 计算actor覆盖率得分 (max_reward: 0.05)
    # 计算parsed_list中不同的actor占can_actors的比例
    # if len(can_actors) > 0:
    #     unique_actors = set(flattened_actors)
    #     valid_unique_actors = unique_actors & set(can_actors)  # 只计算有效的actor
    #     coverage_ratio = len(valid_unique_actors) / len(can_actors)
    #     coverage_score = min(coverage_ratio * 0.05, 0.05)
    # else:
    #     coverage_score = 0.0
    
    # 1.5. 计算每种 actor 类型的占比
    actor_type_score = 0.1
    actor_type = {}
    type_lis = ['rsl',"mac","chess","link","fast"]
    for actor in can_actors:
        for atype in type_lis:
            if str(actor).lower().startswith(atype):
                actor_type[atype] = False
                break
    for actor in flattened_actors:
        for atype in type_lis:
            if str(actor).lower().startswith(atype):
                actor_type[atype] = True
                break
    type_score = len([val for key, val in actor_type.items() if val]) / len(actor_type.keys()) * actor_type_score


    # 2. 计算parsed_list长度得分 (max_reward: 0.2)
    # 假设理想长度在3-10之间
    list_length = len(flattened_actors)
    if 3 <= list_length <= 6:
        # 理想长度范围，给满分
        length_score =  0.2
    elif list_length > 6:
        # 超过10个，分数递减
        length_score = max(0.2 - (list_length - 6) * 0.02, 0.0)
    else:
        # 少于3个，分数按比例递减
        length_score = list_length / 3 * 0.2
    
    # # 3. 检查是否使用嵌套列表 (max_reward: 0.05)
    # nested_score = 0.05 if has_nested_list(parsed_list) else 0.0
    
    # 计算总分
    # total_score = coverage_score + length_score + nested_score + type_score
    total_score = length_score +  type_score

    return total_score



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

def cal_final_score(score_lis: List[float]) -> float:
    score_lis = [x for x in score_lis if x > -0.25] # 排除无法执行出错的 SQL
    all_len = len(score_lis)
    valid_num = len([x for x in score_lis if x > 2.5])
    invalid_num = all_len - valid_num
        
    if valid_num == 0:
        return -0.5
        
    if valid_num > invalid_num:
        return 3.0
        
    if valid_num == invalid_num:
        # 如果正确样本和错误样本数量相同，那么选出正确样本的最小概率是 0.5
        return 3.0 if random.random() < 0.5 else -0.5
        
    prob =  0.3 * valid_num / all_len
    return 3.0 if random.random() < prob else -0.5


def compute_score_batch(batch_responses: dict, batch_size: int = 2):
    # port = 6535  # SquRL
    port = 6517  # FinalRL

    FORMAT_REWARD = 0.5
    use_actor_reward = True
    VALID_SCORE_THRESHOLD = 2.5  # 用于判断是否为有效分数的阈值
    
    # 最终结果字典，存储每个index的分数
    result_scores = {}
    actor_scores = {}
    # 存储验证通过的数据，用于后续批量请求
    # 格式: {instance_id: [(index, task_lis), ...]}
    valid_data = {}
    diff_actor_dict = {}
    # 记录每个 index 属于哪个 instance_id
    index_to_instance = {}
    # 第一步：验证所有response_str，并初始化分数
    for instance_id, response_list in batch_responses.items():
        valid_data[instance_id] = []
        
        for index, response_str, can_actors in response_list:
            # 记录 index 到 instance_id 的映射
            index_to_instance[index] = instance_id
            
            # 调用验证函数
            is_valid, task_lis = validate_response_str(response_str, can_actors)
            
            if is_valid:
                # 验证通过，初始分数为 FORMAT_REWARD
                result_scores[index] = FORMAT_REWARD
                actor_score = compute_actor_score(task_lis, can_actors)
                actor_scores[index] = actor_score
                # 保存用于后续批量请求
                valid_data[instance_id].append((index, task_lis))
            else:
                # 验证失败，分数为0，不参与后续计算
                result_scores[index] = -abs(FORMAT_REWARD)

        # 计算不同的 actor_lis 数量，并赋值给 differ_actor_len
        # 将嵌套列表转换为可哈希的元组形式，用于去重比较
        def list_to_tuple(lst):
            """递归将嵌套列表转换为嵌套元组（可哈希）"""
            if isinstance(lst, list):
                return tuple(list_to_tuple(item) for item in lst)
            return lst
        
        # 收集该 instance_id 下所有验证通过的 task_lis 并去重
        unique_task_tuples = set()
        for index, task_lis in valid_data[instance_id]:
            task_tuple = list_to_tuple(task_lis)
            unique_task_tuples.add(task_tuple)
        
        differ_actor_len = len(unique_task_tuples)
        diff_actor_dict[instance_id] = 2 * len(response_list) - differ_actor_len


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
            # url = f"http://127.0.0.1:{port}/api/run_batch_new"
            url = f"http://127.0.0.1:{port}/api/run_batch"
            status, batch_scores  = http_post_json(url, payload, timeout=1500)
            # response = requests.post(url, json=payload, timeout=300)
            print(batch_scores)
            
            # 还原每个index的分数
            for instance_id in current_batch_ids:
                score_list = batch_scores.get(instance_id, [])
                index_task_pairs = valid_data[instance_id]
                
                # 确保返回的分数列表长度与输入一致
                if len(score_list) == len(index_task_pairs):
                    for (index, _), score in zip(index_task_pairs, score_list):
                        # if use_actor_reward and score >= 2.5:
                        #     score += actor_scores.get(index,0)
                        # 将评测分数加到已有的FORMAT_REWARD上
                        result_scores[index] += score
                else:
                    print(f"[Warning] Score list length mismatch for instance_id: {instance_id}")
                    
        except Exception as e:
            print(f"[Error] Failed to process batch {batch_start}-{batch_end}: {str(e)}")
            # 请求失败时，这些index保持原有的FORMAT_REWARD分数
    
    # 第三步：统计每个 instance_id 的 valid_num（score > VALID_SCORE_THRESHOLD）并计算 group_scalar
    MIN_GROUP_SCALAR = 0.2  # 最小 group_scalar 值，避免为 0
    valid_num_dict = {}  # {instance_id: valid_num}
    for instance_id in batch_responses.keys():
        valid_num_dict[instance_id] = 0
    
    # 统计每个 instance_id 中 score > VALID_SCORE_THRESHOLD 的数量
    for index, score in result_scores.items():
        instance_id = index_to_instance.get(index)
        if instance_id is not None and score > VALID_SCORE_THRESHOLD:
            valid_num_dict[instance_id] += 1
    
    # 计算每个 instance_id 的 group_scalar = valid_num / (2N - Ndiff)
    group_scalar_by_instance = {}
    for instance_id in batch_responses.keys():
        valid_num = valid_num_dict[instance_id]
        denominator = diff_actor_dict.get(instance_id, 1)  # 避免除零
        if denominator <= 0:
            denominator = 1
        # 计算 group_scalar，确保最小值不低于 MIN_GROUP_SCALAR
        group_scalar = valid_num / denominator
        # group_scalar_by_instance[instance_id] = max(group_scalar, MIN_GROUP_SCALAR)
        group_scalar_by_instance[instance_id] = 0.5 + group_scalar

    # 为每个 index 分配对应的 group_scalar
    group_scalar_dict = {}
    for index in result_scores.keys():
        instance_id = index_to_instance.get(index)
        if instance_id is not None:
            group_scalar_dict[index] = group_scalar_by_instance[instance_id]
        else:
            group_scalar_dict[index] = 1.0  # 默认值
    
    # 计算每个 instance_id 的 final_score（用于日志记录）
    final_score_by_instance = {}
    for instance_id in batch_responses.keys():
        # 收集该 instance_id 下所有 index 的 score
        score_lis = []
        for index, score in result_scores.items():
            if index_to_instance.get(index) == instance_id:
                score_lis.append(score)
        
        # 调用 cal_final_score 计算该 instance_id 的最终标量值
        if len(score_lis) > 0:
            final_score_by_instance[instance_id] = cal_final_score(score_lis)
        else:
            final_score_by_instance[instance_id] = -0.5
    
    # 为每个 index 分配对应的 final_score（与 group_scalar_dict 方式一致）
    final_score_dict = {}
    for index in result_scores.keys():
        instance_id = index_to_instance.get(index)
        if instance_id is not None:
            final_score_dict[index] = final_score_by_instance[instance_id]
        else:
            final_score_dict[index] = -0.5  # 默认值
    
    return result_scores, group_scalar_dict, final_score_dict