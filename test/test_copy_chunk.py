# @Author john-y
# @Description TODO
# @Date 2024/3/31 10:24
# @Version 1.0
import json
import os
import time
from multiprocessing import Process, Queue, cpu_count
from typing import List
from os.path import dirname, basename, splitext, join

from src.feature_extraction.database_loader import load_dataset
from src.feature_extraction.extract_features import feature_extractor, get_alias2table, get_plan, PlanInSeq
from src.feature_extraction.node_features import class2json
from src.feature_extraction.plan_features import plan2seq
from src.feature_extraction.sample_bitmap import TreeNode, recover_tree, get_bitmap, prepare_samples, bitand
from src.plan_encoding.meta_info import prepare_dataset
from sync_timer import sync_timed


def partition(data: List,
              chunk_size: int) -> List:
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def add_sample_bitmap_chunk(data : List[str], dst, pid, dataset, sample, sample_num):
  """
  复制文件的一部分。

  参数:
  q (Queue): 包含(start, size)元组的队列，指定要复制的文件块的起始位置和大小。
  src (str): 源文件路径。
  dst (str): 目标文件路径。
  """
  dn, bn = dirname(dst), basename(dst)
  fn, ext = splitext(bn)
  rfn = fn + "-" + str(pid) + ext
  out_path = join(dn, rfn)
  print("out_path : ", out_path)
  with open(out_path, 'w') as f_dst:  # 以读写模式打开目标文件
      for plan in data:
        parsed_plan = json.loads(plan)
        nodes_with_sample = []
        for node in parsed_plan['seq']:
            bitmap_filter = []
            bitmap_index = []
            bitmap_other = []
            if node != None and 'condition' in node:
                predicates = node['condition']
                if len(predicates) > 0:
                    root = TreeNode(predicates[0], None)
                    if len(predicates) > 1:
                        recover_tree(predicates[1:], root)
                    bitmap_other = get_bitmap(root, data, sample, sample_num)
            if node != None and 'condition_filter' in node:
                predicates = node['condition_filter']
                if len(predicates) > 0:
                    root = TreeNode(predicates[0], None)
                    if len(predicates) > 1:
                        recover_tree(predicates[1:], root)
                    bitmap_filter = get_bitmap(root, data, sample, sample_num)
            if node != None and 'condition_index' in node:
                predicates = node['condition_index']
                if len(predicates) > 0:
                    root = TreeNode(predicates[0], None)
                    if len(predicates) > 1:
                        recover_tree(predicates[1:], root)
                    bitmap_index = get_bitmap(root, data, sample, sample_num)
            if len(bitmap_filter) > 0 or len(bitmap_index) > 0 or len(bitmap_other) > 0:
                bitmap = [1 for _ in range(sample_num)]
                bitmap = bitand(bitmap, bitmap_filter)
                bitmap = bitand(bitmap, bitmap_index)
                bitmap = bitand(bitmap, bitmap_other)
                node['bitmap'] = ''.join([str(x) for x in bitmap])
            nodes_with_sample.append(node)
        parsed_plan['seq'] = nodes_with_sample
        f_dst.write(json.dumps(parsed_plan))
        f_dst.write('\n')

# @sync_timed()
def add_sample_bitmap_mp(src, dst, dataset, sample, sample_num, num_processes = cpu_count()):
  """
  使用多进程复制文件。

  参数:
  src (str): 源文件路径。
  dst (str): 目标文件路径。
  num_processes (int): 使用的进程数。
  """
  print(f"num_processes: {num_processes}")
  for pid in range(num_processes):
      dn, bn = dirname(src), basename(src)
      fn, ext = splitext(bn)
      rfn = fn + "-" + str(pid)  + ext
      a_out_path = join(dn, rfn)
      print("src : ", src)

      processes = []
      with open(src, 'r') as ff:
        contents = ff.readlines()
        process = Process(target=add_sample_bitmap_chunk, args=(contents, dst, pid, dataset, sample, sample_num))
        processes.append(process)

  # 创建并启动进程
  for p in processes:
    p.start()

  # 等待所有进程完成
  for p in processes:
    p.join()

def feature_extractor_chunk(chunk, out_path, pid):
    dn, bn = dirname(out_path), basename(out_path)
    fn, ext = splitext(bn)
    rfn = fn + "-" + str(pid) + ext
    a_out_path = join(dn, rfn)
    print("a_out_path : ", a_out_path)
    with open(a_out_path, 'w') as f:
        for plan in chunk:
            if plan != 'null\n':
                plan = json.loads(plan)[0]['Plan']
                if plan['Node Type'] == 'Aggregate':
                    plan = plan['Plans'][0]
                alias2table = {}
                get_alias2table(plan, alias2table)
                subplan, cost, cardinality = get_plan(plan)
                seq, _ = plan2seq(subplan, alias2table)
                seqs = PlanInSeq(seq, cost, cardinality)

                f.write(class2json(seqs) + '\n')

def feature_extractor_mp(input_path, out_path, num_processes = cpu_count()):
    with open(input_path, 'r') as f:
        contents = f.readlines()
        length = len(contents)
        chunk_size = length // num_processes
        print(f"input_path length : {length}, chunk_size: {chunk_size}")
        pid = 0
        processes = []
        for chunk in partition(contents, chunk_size):
            process = Process(target=feature_extractor_chunk, args=(chunk, out_path, pid))
            processes.append(process)
            pid = pid + 1

        # 创建并启动进程
        for p in processes:
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

# 示例使用
if __name__ == '__main__':
    t1 = time.time()

    dataset = load_dataset('/Users/john-y/Downloads/test_files_open_source/imdb_data_csv')
    column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, table_names = prepare_dataset(dataset)
    sample_num = 1000
    sample = prepare_samples(dataset, sample_num, table_names)

    t2 = time.time()
    print(f'1. time cost: {(t2 - t1):.4f} seconds')
    feature_extractor_mp('/Users/john-y/Downloads/test_files_open_source/plans.json',
                      '/Users/john-y/Downloads/test_files_open_source/plans_seq.json')

    add_sample_bitmap_mp('/Users/john-y/Downloads/test_files_open_source/plans_seq.json',
                          '/Users/john-y/Downloads/test_files_open_source/test_copy_chunk_process.json', dataset, sample, sample_num)
    t3 = time.time()
    print(f'2. time cost: {(t3 - t2):.4f} seconds')