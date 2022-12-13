import time
from multiprocessing import Pool
from functools import partial
import argparse
from multiprocessing import cpu_count
from pdfplumber.utils import cluster_objects
from operator import itemgetter
import pdfplumber
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import re
import copy


def get_current_bulk_content(page):
    if page is None:
        return []
    content = page.extract_text()
    content = content.split('\n')
    return content


def read_bulk_data_for_pure_text(path, bulks, cur_slice):
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        size = len(pages)
        if size < bulks:
            print('total page is less than bulk, directly every process deal with each page')
            contents = []
            if cur_slice > size:
                return 'none data has been processed'
            else:
                content = get_current_bulk_content(pages[cur_slice])
                contents.append(content)
                return contents
        containers_per_bulk = size // bulks
        contents = []
        start = (cur_slice - 1) * containers_per_bulk
        if cur_slice == bulks:
            end = size
        else:
            end = cur_slice * containers_per_bulk
        for i in range(start, end):
            page = pages[i]
            if page is None:
                contents.append([])
                continue
            content = page.extract_text()
            tmp = content.split('\n')
            contents.append(tmp)
    return contents


def process_number(parallel_num, path):
    cpu_kernel = cpu_count()
    available_kernel = min(cpu_kernel // 2, cpu_kernel - 2)
    if parallel_num != 1:
        if isinstance(parallel_num, int):
            if available_kernel > parallel_num:
                available_kernel = parallel_num
    f = Path(path)
    size = f.stat().st_size
    size = round(size / 1024 / 1024, 3)
    if size < 4:
        advice_number = 4
    else:
        advice_number = 8
    process_kernel = min(advice_number, available_kernel)
    return process_kernel


def parallel_process(path, parallel_num=1, parse_type=1):
    process_kernel = process_number(parallel_num, path)
    print('current process:', process_kernel)
    pool = Pool(processes=process_kernel)
    if parse_type == 1:
        parse_file = partial(read_bulk_data_for_pure_text, path)
    else:
        parse_file = partial(read_bulk_data_for_various_information, path)
    parse_bulk = partial(parse_file, process_kernel)
    slice_number = [i + 1 for i in range(process_kernel)]
    bulk_results = pool.map(parse_bulk, slice_number)
    pool.close()
    pool.join()
    if parse_type == 1:
        final_result = []
        for bulk_result in bulk_results:
            final_result.extend(bulk_result)
        return final_result
    else:
        contents, leftsize, rightsize, chars2gap = merge_multi_process_bulk_result(bulk_results)
        return contents, leftsize, rightsize, chars2gap


def merge_multi_process_bulk_result(total_results):
    contents, leftsize, rightsize, chars2gap = [], defaultdict(list), defaultdict(list), dict()
    for bulk_result in total_results:
        bulk_contents, bulk_leftsize, bulk_rightsize, bulk_chars2gap = bulk_result
        contents.extend(bulk_contents)
        for k, v in bulk_leftsize.items():
            if k not in leftsize:
                leftsize[k] = v
            else:
                leftsize[k].extend(v)
        for k, v in bulk_rightsize.items():
            if k not in rightsize:
                rightsize[k] = v
            else:
                rightsize[k].extend(v)
        for k, v in bulk_chars2gap.items():
            if k not in chars2gap:
                chars2gap[k] = v
            else:
                chars2gap[k] = max(chars2gap[k], v)
    for k, v in leftsize.items():
        v.sort()
    for k, v in rightsize.items():
        v.sort(reverse=True)
    return contents, leftsize, rightsize, chars2gap


def read_bulk_data_for_various_information(path, bulks, cur_slice):
    tolerance_y = 3
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        size = len(pages)
        leftsize = defaultdict(list)
        rightsize = defaultdict(list)
        chars2gap = dict()
        if size < bulks:
            print('total page is less than bulk, directly every process deal with each page')
            contents = []
            if cur_slice > size:
                return 'none data has been processed'
            else:
                cur_page = pages[cur_slice]
                every_page_contents = get_each_page_data(cur_page, tolerance_y, leftsize, rightsize, chars2gap)
                contents.append(every_page_contents)
                return contents, leftsize, rightsize, chars2gap
        containers_per_bulk = size // bulks
        contents = []
        start = (cur_slice - 1) * containers_per_bulk
        if cur_slice == bulks:
            end = size
        else:
            end = cur_slice * containers_per_bulk
        for i in range(start, end):
            cur_page = pages[i]
            every_page_contents = get_each_page_data(cur_page, tolerance_y, leftsize, rightsize, chars2gap)
            contents.append(every_page_contents)

    return contents, leftsize, rightsize, chars2gap


def get_each_page_data(cur_page, tolerance_y, leftsize, rightsize, chars2gap):
    every_page_contents = []
    chars = cur_page.chars
    doctop_clusters = cluster_objects(chars, "doctop", tolerance_y)
    get_last_first_char_information(doctop_clusters, leftsize, rightsize, every_page_contents, chars2gap)
    return every_page_contents


def get_last_first_char_information(doctop_clusters, leftsize, rightsize, every_page_contents, chars2gap):
    lines_number = len(doctop_clusters)
    for i in range(lines_number):
        line = doctop_clusters[i]
        first_token = line[0]
        first_char_size = round(first_token['size'], 2)
        first_position = first_token['x0']
        leftsize[first_char_size].append(first_position)
        last_token = line[-1]
        last_char_size = round(last_token['size'], 2)
        last_position = last_token['x0']
        rightsize[last_char_size].append(last_position)
        line_information = get_every_line_information(line, chars2gap)
        line_information['first_char_position'] = round(first_position, 2)
        line_information['last_char_position'] = round(last_position, 2)
        every_page_contents.append(line_information)


def get_every_line_information(line, chars2gap):
    char_size = []
    text = []
    total_char = len(line)
    for i in range(total_char):
        char = line[i]
        size = round(char['size'], 2)
        char_size.append(size)
        text.append(char['text'])
    size2num = Counter(char_size)
    if len(size2num) > 1:
        print('the number of size of char in a line is more than one')
    size_frequency = sorted(size2num.items(), key=lambda x: x[1], reverse=True)
    mode = size_frequency[0][0]
    line_information = {'text': ''.join(text), 'mode_size': mode}
    if mode not in chars2gap:
        gap = 0.000
        for i in range(total_char):
            char = line[i]
            if len(line) > 5:
                if i > 0:
                    pre_char = line[i - 1]
                    gap = round(max(char['x0'] - pre_char['x0'], gap), 3)
        chars2gap[mode] = gap
    return line_information


def judge_new_line(previous, current, leftsize, rightsize, chars2gap):
    if not previous:
        return True
    special_tokens = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', '']
    if previous['mode_size'] != current['mode_size']:
        return True
    if current['mode_size'] > 12:
        return True
    first_position = current['first_char_position']
    last_position = previous['last_char_position']
    size = previous['mode_size']
    text_distribute_right_position = rightsize[size][0]
    text_distribute_left_position = leftsize[size][0]
    previous_text = previous['text']
    current_text = current['text']
    if abs(last_position - text_distribute_right_position) > chars2gap[size] + 0.001:
        return True  # to do
    if abs(last_position - text_distribute_right_position) < chars2gap[size] + 0.001 < \
            abs(first_position - text_distribute_left_position):
        # the position of last char in previous line is rightest, but the position of first char in current line
        # is not leftest
        new_line_flag = False
        if previous_text[0] in special_tokens and current_text[0] not in special_tokens:
            return False
        return True
    # when the paragraph is catalogue,the first char is leftest and the last char is rightest.
    s_p_2 = '([\.]+[\d]+$)'
    Iscatalogue = get_re_format1_result(previous_text, s_p_2)
    if Iscatalogue:
        return True

    return False


def merge_different_line2paragraph(all_pages_contents, leftsize, rightsize, chars2gap):
    paragraphs = []
    page_number = len(all_pages_contents)
    for i in range(page_number):
        page = all_pages_contents[i]
        line_number = len(page)
        for j in range(line_number - 1):
            if i == 0 and j == 0:
                paragraphs.append(page[j]['text'])
                continue
            if i != 0 and j == 0:
                if len(all_pages_contents[i - 1]) < 2:
                    previous = None
                else:
                    previous = all_pages_contents[i - 1][-2]
            else:
                previous = page[j - 1]
            current = page[j]
            new_line_flag = judge_new_line(previous, current, leftsize, rightsize, chars2gap)
            if new_line_flag:
                paragraphs.append(current['text'])
                continue

            merge_line = paragraphs.pop() + current['text']
            paragraphs.append(merge_line)
    return paragraphs


def get_re_format1_result(st, s_p):
    pattern = re.compile(s_p)
    search = re.search(pattern, st)
    if search:
        return True
    return False


def filter_page_number(st):
    s_p_1 = '(第[\d]+页)'
    has_match = get_re_format1_result(st, s_p_1)
    s_p_2 = '([\.]+[\d]+$)'

    return has_match


def merge_different_line2paragraph_from_different_results(pure_texts, position_texts, leftsize, rightsize, chars2gap):
    paragraphs_information = []
    page_number = len(pure_texts)
    for i in range(page_number):
        current_page_pure = pure_texts[i]
        current_page_position = position_texts[i]
        line_number = len(current_page_pure)
        for j in range(line_number):
            current_line_new = copy.deepcopy(current_page_position[j])
            del current_line_new['text']
            if i == 0 and j == 0:
                current_line_new['text'] = current_page_pure[j]
                paragraphs_information.append(current_line_new)
                continue
            if i != 0 and j == 0:
                text = current_page_position[j]['text']
                is_page_number = filter_page_number(text)
                if is_page_number:
                    continue
                if len(position_texts[i - 1]) == 1:
                    # previous page has no pure text, except page information
                    is_page_number = filter_page_number(position_texts[i - 1][-1]['text'])
                    if is_page_number:
                        previous = None
                    else:
                        previous = paragraphs_information[-1]
                else:
                    previous = paragraphs_information[-1]
            else:
                text = current_page_position[j]['text']
                if isinstance(text, str):
                    is_page_number = filter_page_number(current_page_position[j]['text'])
                    if is_page_number:
                        continue
                else:
                    print(text)
                previous = paragraphs_information[-1]
            current_line = current_page_position[j]

            new_line_flag = judge_new_line(previous, current_line, leftsize, rightsize, chars2gap)
            if new_line_flag:
                current_line_new['text'] = current_page_pure[j]
                paragraphs_information.append(current_line_new)
                continue
            merge_text = paragraphs_information.pop()['text'] + current_page_pure[j]
            current_line_new['text'] = merge_text
            paragraphs_information.append(current_line_new)

    return paragraphs_information


def get_different_parse_result_by_line(path, parallel):
    parse_file = partial(parallel_process, path)
    parse_bulk = partial(parse_file, parallel)
    executor = ThreadPoolExecutor(max_workers=2)
    task1 = executor.submit(lambda x: parse_bulk(*x), [1])
    task2 = executor.submit(lambda x: parse_bulk(*x), [2])
    tasks = [task1, task2]
    wait(tasks, return_when=ALL_COMPLETED)
    task_results = []
    for task in tasks:
        res = task.result()
        task_results.append(res)
    return task_results


def get_paragraph_information_from_parse_result(path, parallel):
    """
    this function is used to parse document the type of which is pdf, get the pure text from pdf,
    :param path:
    :param parallel:
    :return:
    """
    task_results = get_different_parse_result_by_line(path, parallel)
    results01 = task_results[0]
    results02 = task_results[1]
    assert len(results01) == len(results02[0])
    every_line = [len(results01[i]) == len(results02[0][i]) for i in range(len(results01))]
    assert all(every_line)
    paragraphs_information = merge_different_line2paragraph_from_different_results(results01, results02[0],
                                                                                   results02[1], results02[2],
                                                                                   results02[3])

    return paragraphs_information


def main(source, destination, parallel=1):
    paragraphs = get_paragraph_information_from_parse_result(source, parallel)
    with open(destination, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(paragraph['text'])
            f.write('\n')


if __name__ == '__main__':
    filepath = 'D:/project/python/1102/ikm_data/auto_doc_info_extractor/parse_pdf/redBook/iuap-AI工作坊红皮书.pdf'
    filepath = '../redBook/iuap-DevOps云上调试红皮书.pdf'


    begin_time = time.time()
    paragraphs = get_paragraph_information_from_parse_result(filepath, 1)
    print('multi task time: ', time.time() - begin_time)

    # paragraphs = get_text_from_pdf(filepath)
    destination = 'iuap-DevOps云上调试红皮书1212b.' + 'txt'
    with open(destination, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(paragraph['text'])
            f.write('\n')
