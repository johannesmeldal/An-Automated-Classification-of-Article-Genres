import csv

def move_train_test_rows(input_file, output_train_file, output_test_file, total_articles, proportion_train, proportion_test):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        train_rows = []
        test_rows = []

        for row in reader:
            if row[1] == 'TRAIN':
                train_rows.append(row)
            elif row[1] == 'TEST':
                test_rows.append(row)

    num_train_rows = min(int(len(train_rows) * proportion_train), total_articles // 2)
    num_test_rows = min(int(len(test_rows) * proportion_test), total_articles // 2)

    with open(output_train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(train_rows[:num_train_rows])

    with open(output_test_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(test_rows[:num_test_rows])

# Move a proportion of train and test rows to separate files
move_train_test_rows('article_data.csv', 'sample_train_data.csv', 'sample_test_data.csv', total_articles=1000, proportion_train=0.5, proportion_test=0.5)
