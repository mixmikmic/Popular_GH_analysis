get_ipython().system('wget http://rest.kegg.jp/get/ec:5.4.2.2 -O ec_5.4.2.2.txt')

from Bio.KEGG import Enzyme

records = Enzyme.parse(open("ec_5.4.2.2.txt"))

record = list(records)[0]

record.classname

record.entry

from Bio.KEGG import REST

from Bio.KEGG import Enzyme

request = REST.kegg_get("ec:5.4.2.2")

open("ec_5.4.2.2.txt", 'w').write(request.read().decode("utf-8"))

records = Enzyme.parse(open("ec_5.4.2.2.txt"))

record = list(records)[0]

record.classname

record.entry

from Bio.KEGG import REST

human_pathways = REST.kegg_list("pathway", "hsa").read()

human_pathways.decode("utf-8").split("\n")[0:5]

# Filter all human pathways for repair pathways
repair_pathways = []
for line in human_pathways.decode("utf-8").rstrip().split("\n"):
    entry, description = line.split("\t")
    if "repair" in description:
        repair_pathways.append(entry)

repair_pathways

# Get the genes for pathways and add them to a list
repair_genes = [] 
for pathway in repair_pathways:
    pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway

    # iterate through each KEGG pathway file, keeping track of which section
    # of the file we're in, only read the gene in each pathway
    current_section = None
    for line in pathway_file.decode("utf-8").rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section
        
        if current_section == "GENE":
            gene_identifiers, gene_description = line[12:].split("; ")
            gene_id, gene_symbol = gene_identifiers.split()

            if not gene_symbol in repair_genes:
                repair_genes.append(gene_symbol)

print("There are %d repair pathways and %d repair genes. The genes are:" %         (len(repair_pathways), len(repair_genes)))
print(", ".join(repair_genes))



