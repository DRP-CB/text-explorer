import docx
import glob

glob.os.mkdir('extracted_text')

def make_filename(path):
    return glob.os.path.normpath(path).split(glob.os.sep)[-1].replace('.docx', ".txt")

def extract_text(file_path):
    document = docx.Document(file_path)
    doctext = ' '.join(
        paragraph.text.replace('\xa0',' ') for paragraph in document.paragraphs
    )
    with open('extracted_text/' + make_filename(file_path), 'w') as f:
        f.write(doctext)
        
files = glob.glob(glob.os.getcwd() +'/' +'*.{}'.format('docx'))


for file in files :
    extract_text(file)