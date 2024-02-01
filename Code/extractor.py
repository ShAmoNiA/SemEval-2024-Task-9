from bs4 import BeautifulSoup
import requests
import csv
import os

def parser(name,question,answer):
    questionHold = []
    answerHold = []
    with open('riddles/'+name+'.html', 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        
        questions = soup.find_all(class_=question)
        for element in questions:
            questionHold.append(element.get_text(strip=True).replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace('\n',''))
            
        answers = soup.find_all(class_=answer)
        for element in answers:
            answerHold.append(element.get_text(strip=True).split('Answer:')[1].replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace('\n',''))
            
    return [questionHold,answerHold]
            
def downloader(url,ScdPart_url,count):
    print(url+ScdPart_url)
    response = requests.get(url+ScdPart_url)
    if response.status_code == 200:
        with open('riddles/'+url.split('/')[-2]+str(count)+'.html', 'w', encoding='utf-8') as file:
            file.write(response.text)
        return True
    else:
        return False
    
def csv_write(csv_file_path, data_to_append):
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data_to_append)
    else:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            if os.stat(csv_file_path).st_size == 0:
                header = ['questions', 'answers']  
                csv_writer.writerow(header)
            csv_writer.writerow(data_to_append)


urls = ["https://www.ba-bamail.com/riddles/hardest-riddles/","https://www.ba-bamail.com/riddles/dirty-funny-riddles/",
        "https://www.ba-bamail.com/riddles/math-riddles/","https://www.ba-bamail.com/riddles/logic-riddles/",
        "https://www.ba-bamail.com/riddles/logic-riddles/","https://www.ba-bamail.com/riddles/riddles-for-kids/",
        "https://www.ba-bamail.com/riddles/word-riddles/","https://www.ba-bamail.com/riddles/easy-riddles/",
        "https://www.ba-bamail.com/riddles/tricky-riddles/","https://www.ba-bamail.com/riddles/hard-riddles/",
        "https://www.ba-bamail.com/riddles/hardest-riddles/","https://www.ba-bamail.com/riddles/who-am-i-riddles/",
        "https://www.ba-bamail.com/riddles/what-am-i-riddles/","https://www.ba-bamail.com/riddles/food-riddles/",
        "https://www.ba-bamail.com/riddles/science-riddles/","https://www.ba-bamail.com/riddles/sports-riddles/","https://www.ba-bamail.com/riddles/funny-riddles/",
        "https://www.ba-bamail.com/riddles/what-is-it/","https://www.ba-bamail.com/riddles/mystery-solving/",
        "https://www.ba-bamail.com/riddles/animal-riddles/","https://www.ba-bamail.com/riddles/love-riddles/",
        "https://www.ba-bamail.com/riddles/nature-riddles/"
        ]
question = 'tm-riddle-box-question margin-bottom-20'
answer = 'tm-riddle-answer-show' 

for url in urls:
    count = 0 
    csv_file = url.split('/')[-2]+'.csv'
    while(True):
        ScdPart_url = "/?skip="+ str(30*count)
        if(downloader(url,ScdPart_url,count) == False):
            break
        ListHold = parser(url.split('/')[-2]+str(count),question,answer)
        length = len(ListHold[0])
        if length == 0:
            break
        for i in range(length):
            csv_write(csv_file,[ListHold[0][i],ListHold[1][i]])
        count +=1
    