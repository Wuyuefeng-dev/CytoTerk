import nbformat as nbf
import re
import os

with open("demo_analysis.py", "r") as f:
    source = f.read()

# Replace show=False with show=True
source = source.replace("show=False", "show=True")
# Comment out savefig and plt.close but add pass to prevent empty blocks
source = re.sub(r'([ \t]*)(.*?\.savefig\()', r'\1pass # \2', source)
source = source.replace("plt.close()", "pass # plt.close()")

# Remove markdown file stuff
source = re.sub(r'([ \t]*)md_file\.', r'\1pass # md_file.', source)
source = source.replace('md_file = open("demo_figs/demo_report.md", "w")', '# md_file = open(...)')

# Extract imports
imports_part = source.split('def main():')[0].strip()

# Change input() to automatically use choice 1 (pbmc3k)
source = source.replace("choice = input(\"Enter 1 or 2 [default: 1]: \").strip()", "choice = '1'")
source = source.replace("return", "sys.exit(1)")

# Extract main body
main_part = source.split('def main():')[1].split('if __name__ == "__main__":')[0]

# Split main body by major sections to create cells
sections = re.split(r'\n(    )# (\d+\..*?)\n', main_part)

nb = nbf.v4.new_notebook()
cells = []

# First cell: imports + matplotlib inline
first_cell = "%matplotlib inline\nimport sys\nsys.path.append('src')\n" + imports_part
cells.append(nbf.v4.new_code_cell(first_cell))

# Process Sections
current_code = ""
for i in range(1, len(sections), 3):
    indent = sections[i]
    title = sections[i+1]
    code_block = sections[i+2]
    
    # Unindent code_block by 4 spaces
    unindented_code = []
    for line in code_block.split('\n'):
        if line.startswith('    '):
            unindented_code.append(line[4:])
        elif line == '':
            unindented_code.append('')
        else:
            unindented_code.append(line)
            
    clean_code = '\n'.join(unindented_code).strip()
    
    # Add title as markdown
    cells.append(nbf.v4.new_markdown_cell("## " + title))
    cells.append(nbf.v4.new_code_cell(clean_code))

nb['cells'] = cells

with open('demo_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("demo_analysis.ipynb generated successfully from demo_analysis.py!")
