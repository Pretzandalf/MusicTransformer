# Use a pipeline as a high-level helper
from transformers import pipeline
import argparse
import requests
#print('в какую папку сохранить txt файл:')
#path = str(input())


pipe = pipeline("text-generation", model="ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16")


order = ["Around the World", 'purple stain', 'californication']

url_git = 'https://raw.githubusercontent.com/Pretzandalf/MusikTransformer/main/Prompt'



def LammaTest(path):
    for i in range(1, len(order) + 1):

        prompt = requests.get(url_git + i +'.txt')

        for num in range(1, 4):

            res = pipe(prompt)

            with open((path + 'Lamma70B.txt'), "a", encoding="utf-8") as file:
                file.write('песня: ' + order[i] + 'запуск '+ num + ':' + res + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLAMA test')
    parser.add_argument('-s', '--save_dir', default=None, type=str,
                      help='path to save directory')
    args = parser.parse_args()
    LammaTest(args.save_dir)
