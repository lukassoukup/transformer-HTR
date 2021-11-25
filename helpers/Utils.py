
def parse_authors(dataset_input):
    datasets = list()
    if dataset_input.isnumeric():
        datasets.append(f'author_{dataset_input}')
    else:
        start, end = dataset_input.split('-')
        for i in range(int(start), int(end) + 1):
            datasets.append(f'author_{i}')

    return datasets


def read_label_file(file_path, subdir, dataset):
    file_label = list()

    with open(file_path, 'r', encoding='utf-8') as f_tr:
        for line in f_tr.readlines():
            line = [x.strip() for x in line.split()]
            line[0] = f"{dataset}/{subdir}/{line[0]}"
            file_label.append(line)  # = [i.split(' ') for i in data_tr]

    return file_label
