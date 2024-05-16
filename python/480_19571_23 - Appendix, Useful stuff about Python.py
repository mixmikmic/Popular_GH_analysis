handle = open("data/m_cold.fasta", "r")
handle.readline()

from Bio import SeqIO
for record in SeqIO.parse("data/m_cold.fasta", "fasta"):
    print(record.id, len(record))

from Bio import SeqIO
handle = open("data/m_cold.fasta", "r")
for record in SeqIO.parse(handle, "fasta"):
    print(record.id, len(record))
handle.close()

my_info = 'A string\n with multiple lines.'
print(my_info)

from io import StringIO
my_info_handle = StringIO(my_info)
first_line = my_info_handle.readline()
print(first_line)

second_line = my_info_handle.readline()
print(second_line)



