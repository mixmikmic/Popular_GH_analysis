f = open("births.csv", 'r')
text = f.read()
print(text)

lines_list = text.split("\n")
lines_list

data_no_header = lines_list[1:len(lines_list)]
days_counts = dict()

for line in data_no_header:
    split_line = line.split(",")
    day_of_week = split_line[3]
    num_births = int(split_line[4])

    if day_of_week in days_counts:
        days_counts[day_of_week] = days_counts[day_of_week] + num_births
    else:
        days_counts[day_of_week] = num_births

days_counts

