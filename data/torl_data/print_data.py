import os
import pandas as pd
import json
from datetime import datetime


def preview_parquet(path, n_rows=5, save_json=True):
    """
    打印指定路径下 .parquet 文件的前几行，并保存为JSON
    :param path: 文件路径或目录路径
    :param n_rows: 要显示的行数
    :param save_json: 是否保存为JSON文件
    """
    results = {}

    # 判断是否为文件
    if os.path.isfile(path):
        if path.endswith(".parquet"):
            print(f"\n=== File: {path} ===")
            df = pd.read_parquet(path)
            print(df.head(n_rows))

            # 保存数据到results字典
            results[path] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": df.head(n_rows).to_dict(orient='records')
            }
        else:
            print(f"[Skip] Not a parquet file: {path}")
            return

    # 如果是目录，遍历其中所有 parquet 文件
    elif os.path.isdir(path):
        parquet_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".parquet")
        ]
        if not parquet_files:
            print("No .parquet files found in directory.")
            return

        for f in parquet_files:
            print(f"\n=== File: {f} ===")
            df = pd.read_parquet(f)
            print(df.head(n_rows))

            # 保存数据到results字典
            results[f] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": df.head(n_rows).to_dict(orient='records')
            }
    else:
        print(f"Path not found: {path}")
        return

    # 保存为JSON文件
    if save_json and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"parquet_preview_{timestamp}.json"

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n✓ Results saved to: {json_filename}")


if __name__ == "__main__":
    # 直接调用方法，传入参数
    preview_parquet("test.parquet", n_rows=5, save_json=True)