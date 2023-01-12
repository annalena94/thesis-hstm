import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
from absl import app


# This python script scrapes the interview data from https://www.failory.com/interviews
# The repository contains the data produced by this script in the dat folder.

def main(args):
    # STEP 1 - Get all URLS
    # as a workaround for infinite scrolling on the website, the HTML file has been downloaded and is
    # being used to scrape the general data of the startups
    data = open('../../dat/interviews.html', 'r')
    soup = BeautifulSoup(data, 'html.parser')
    interviews = soup.find_all('div', class_='w-dyn-item')
    links = []

    for interview in interviews:
        link = interview.find('a')['href']
        links.append(link)

    # STEP 2 - Get Data
    startup_data_questions = []
    startup_data_general = []

    for link in links:
        # instantiate empty question and answer array for each interview
        questions = []
        answers = []
        URL = link
        startup = link.split('/')[4]
        website = requests.get(URL)
        results = BeautifulSoup(website.content, 'html.parser')

        # get summary and general data
        summary = results.find('p', class_="content-summary").text
        title = results.find('h1', class_='content-h1').text
        date_of_interview = results.find('div', class_='text-block-interview-date').text
        founder = results.find('div', class_='text-block-interview-name').text
        tags = results.find_all('div', class_='text-block-interview-tags')
        category = tags[0].text
        country = tags[1].text
        revenue = tags[2].text
        cause_of_failure = tags[5].text
        outcome = results.find('div', class_="section-interview-failure w-condition-invisible wf-section")
        if outcome is None:
            outcome = 'failure'
            outcome_numeric = 0
        else:
            outcome = 'success'
            outcome_numeric = 1
        links_to_know_more = []

        # get links to know more
        text_blocks = results.find_all('div', class_="content-rich-text w-richtext")
        for block in text_blocks:
            questions_from_block = block.find_all('h2')
            for question in questions_from_block:
                if "Where can we go to learn more?" in question.text:
                    # go through sibling tags (i.e. p tags) of the question until the next h2-tag appears
                    for answer_part in question.next_siblings:
                        if "h2" in answer_part.name:
                            break
                        links_extracted = answer_part.find_all('a')
                        for link in links_extracted:
                            links_to_know_more.append(link['href'])

        # clean up - delete figures and links and linebreaks
        for figure in results.find_all('figure'):
            figure.decompose()
        for link in results.find_all('a'):
            link.replace_with(link.text)
        for linebreak in results.find_all('br'):
            linebreak.decompose()
        text_blocks = results.find_all('div', class_="content-rich-text w-richtext")

        # get question data
        for block in text_blocks:
            # get all questions in the block (normally three)
            questions_from_block = block.find_all('h2')
            for question in questions_from_block:
                # for each question, append the question to the questions array
                questions.append(question.text)
                # go through sibling tags (i.e. p tags) of the question until the next h2-tag appears
                answer = ""
                for answer_part in question.next_siblings:
                    if "h2" in answer_part.name:
                        break
                    answer = answer + " " + answer_part.text
                # delete semicolons from answer to enhance csv file handling
                answer = answer.replace(';', "")
                # append answer to answer array
                answers.append(answer)

        # data without questions
        startup_data_general.append({
            'startup': startup,
            'link': URL,
            'title': title,
            'summary': summary,
            'date_of_interview': date_of_interview,
            'founder': founder,
            'category': category,
            'country': country,
            'cause_of_failure': cause_of_failure,
            'revenue': revenue,
            'outcome': outcome,
            'outcome_numeric': outcome_numeric,
            'links_to_know_more': links_to_know_more,
        })

        # data with questions
        for i in range(len(questions)):
            startup_data_questions.append({
                'startup': startup,
                'question': questions[i],
                'answer': answers[i]
            })

    # STEP 3 - Write data to csv
    fieldnames_startups = startup_data_questions[0].keys()
    with open('../../dat/startup_data_long_format.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames_startups)
        dict_writer.writeheader()
        dict_writer.writerows(startup_data_questions)

    # STEP 4 - Rearrange question format and cleanup question data
    df = pd.DataFrame.from_dict(startup_data_questions)

    # STEP 4.1: Clean up questions in data.question
    # Group questions together that have very similar wording
    # iterate over all questions and manipulate them (in case of different wordings for same question)
    for i in range(len(df.question)):

        # 1. Who are you and what are you currently working on?
        if "Who are you" in df.question[i]:
            df.at[i, "question"] = "Who are you and what are you currently working on?"

        # 2. What’s your background and how did you come up with the idea of xy?
        if "What’s your background and how did you come" in df.question[i] \
                or "What’s your backstory? How did you come up with the idea?" in df.question[i] \
                or "How did you come up Formatically's idea?" in df.question[i]:
            df.at[i, "question"] = "What’s your background and how did you come up with the idea for your startup?"

        # 3. "What's your background, and what are you currently working on?"
        if "What's your background, and what are you currently working on?" in df.question[i] or \
                "can you tell us a bit about your background and what are you currently working on" in df.question[i] \
                or "What are your backgrounds" in df.question[i] \
                or "What's your background and what are you currently working on?" in df.question[i] \
                or "Can you tell us a little bit" in df.question[i]:
            df.at[i, "question"] = "What’s your background, and what are you currently working on?"

        # 3. Which were the causes of xy’s failure? Why did you stop taking xy forward?
        if "Which were the causes" in df.question[i] \
                or "What were the causes of" in df.question[i] \
                or "Which was the cause" in df.question[i] \
                or "What caused the failure of" in df.question[i] \
                or "cause of failure" in df.question[i] \
                or "fail" in df.question[i] \
                or "Why did you stop trying" in df.question[i]:
            df.at[i, "question"] = "Which were the causes of failure?"

        # 4. What was xy about?
        if "What was " in df.question[i] and "about" in df.question[i]:
            df.at[i, "question"] = "What was your startup about?"

        # 5. Which marketing strategies did you use to attract xy first customers?
        # + Which were your marketing strategies to grow your business?
        if "marketing strategies" in df.question[i] or "marketing plans" in df.question[i] \
                or " 'How did you market" in df.question[i]:
            df.at[i, "question"] = "Which marketing strategies did you use?"

        # 6. What motivated you to start xy?
        if "What motivated you to start" in df.question[i] \
                or "What motivated you to build it?" in df.question[i]:
            df.at[i, "question"] = "What motivated you to start your startup?"

        # 7. How did you build xy?
        if "How did you build" in df.question[i] \
                or "how did you build" in df.question[i] \
                or "How did you come up with Lunch Money’s idea and build it?" in df.question[i] \
                or "What went into building the initial product?" in df.question[i] \
                or "Which technologies did you use to build it?" in df.question[i]:
            df.at[i, "question"] = "How did you build it?"

        # 8. How did you grow xy?
        if "How did you grow" in df.question[i] \
                or "strategies to grow your business" in df.question[i]:
            df.at[i, "question"] = "How did you grow it?"

        # 9. Which was the problem with xy? How did you realize?
        if "Which was the problem with" in df.question[i] \
                or "What was the problem with" in df.question[i] \
                or "Which were the problems" in df.question[i]:
            df.at[i, "question"] = "Which was the problem with your product/service? How did you realize?"

        # 10. During the process of building & growing xy, which were the worst mistakes you committed?
        if "During the process of building " in df.question[i] and "worst mistakes" in df.question[i]:
            df.at[i, "question"] = "During the process of building & growing your startup, which were the worst mistakes you committed?"

        # 11. Since starting xy, what have been your main lessons?
        if "Since starting " in df.question[i] and "main lessons" in df.question[i]:
            df.at[i, "question"] = "Since starting your startup, what have been your main lessons?"

        # 12. Questions with recommendations
        if "recommend" in df.question[i]:
            df.at[i, "question"] = "What resources (podcasts, books etc.) do you recommend?"

        # 13. If you had the chance to do things differently, what would you change?
        if "what would you change" in df.question[i] \
                or "If you had the chance of doing only one thing differently" in \
                df.question[i] or "start over" in df.question[i] \
                or "If you had the chance to do things differently" in df.question[i] \
                or "changing one thing" in df.question[i]:
            df.at[i, "question"] = "If you had the chance to do things differently, what would you change?"

        # 14. What were the biggest challenges you faced and obstacles you overcame?"
        if "What were the biggest challenges" in df.question[i]:
            df.at[i, "question"] = "What were the biggest challenges you faced and obstacles you overcame?"

        # 15. How did you come up with the idea to build your startup? What was the motivation behind it?
        if "How did you come up with the idea to build" in df.question[i] and " What was the motivation behind it?" in df.question[i]:
            df.at[i, "question"] = "How did you come up with the idea to build your startup? What was the motivation behind it?"

        # 16. What did eventually cause the shut-down?
        if "shut down" in df.question[i]:
            df.at[i, "question"] = "What did eventually cause the shut down?"

        # 17. Which were your revenues, expenses and/or losses?
        if "revenue" in df.question[i] or "losing" in df.question[i]:
            df.at[i, "question"] = "Which were your revenues, expenses and/or losses?"

        # 18. How are you doing today and what are your goals for the future?
        if "goals" in df.question[i]:
            df.at[i, "question"] = "How are you doing today and what are your goals for the future?"

        # 19. What did you learn and what's your advice for young entrepreneurs?
        if "learned" in df.question[i] or "What did you learn?" in df.question[i] or "advice" in df.question[i] \
                or "If you could talk to your former self" in df.question[i]:
            df.at[i, "question"] = "What did you learn and what's your advice for young entrepreneurs?"

        # 20. Where can we go to learn more?
        if "Where can we go to learn more?" in df.question[i]:
            df.at[i, "question"] = "Where can we go to learn more?"

        # 21. What were your key levers to start growing your startup?
        if "What were your key levers to start growing" in df.question[i]:
            df.at[i, "question"] = "What were your key levers to start growing your startup?"

        # 22. Which were your greatest disadvantages? What were your worst mistakes?
        if "disadvantages" in df.question[i] and "mistakes" in df.question[i]:
            df.at[i, "question"] = "Which were your greatest disadvantages? What were your worst mistakes?"

        # 23. What were the biggest obstacles you overcame? What were your worst mistakes?
        if "obstacles" in df.question[i] and "mistakes" in df.question[i]:
            df.at[i, "question"] = "What were the biggest obstacles you overcame? What were your worst mistakes?"

        # 24. How did you realize the project was not going in the right direction? What did make you finally shut down your project?
        if "How did you realize" in df.question[i] and "right direction" in df.question[i]:
            df.at[
                i, "question"] = "How did you realize the project was not going in the right direction? What did make you finally shut down your project?"

        # 25. What are you favorite entrepreneurial/startup resources?
        if "entrepreneurial resources" in df.question[i] or "startup resources" in df.question[i]:
            df.at[i, "question"] = "What are you favorite entrepreneurial resources?"

    # STEP 4.2: Merge questions with the different wording but similar meaning
    for i in range(len(df.question)):

        # Group 1: Idea - backstory & motivation
        # Q: How did you come up with the idea to build it? What was the main motivation behind it?
        # Q: What’s your background and how did you come up with the idea for your startup?
        # Q: What motivated you to start your startup?
        # Q: Why did you build it?
        # Q: What was your startup about?
        if "with the idea" in df.question[i] or "motivat" in df.question[i] or "why" in df.question[i] or "Why" in \
                df.question[i] or "What was your startup about?" in df.question[i]:
            df.at[i, "question"] = "idea_backstory_motivation"

        # Group 2: Marketing strategies
        # Q: Which marketing strategies did you use?
        # Q: How did you market Raw Gains?
        # Q: Which SEO strategies did you use to grow Waterproof Digital Camera Blog?
        # Q: What channels did you use to find students and teachers?
        if "strategies" in df.question[i] or "market" in df.question[i] or "channels" in df.question[i]:
            df.at[i, "question"] = "marketing_strategies"

        # Group 3: Recommended resources
        # Q: What are you favorite entrepreneurial resources?
        # Q: What resources (podcasts, books etc.) do you recommend?
        if "resource" in df.question[i]:
            df.at[i, "question"] = "recommended_resources"

        # Group 4: Challenges, Obstacles & Mistakes
        # Q: What were your biggest mistakes and challenges you had to overcome?
        # Q: What were the biggest obstacles you overcame?
        # Q: What were the biggest challenges you faced and obstacles you overcame?
        if "mistakes" in df.question[i] or "challenges" in df.question[i] or "obstacles" in df.question[i]:
            df.at[i, "question"] = "challenges_obstacles_mistakes"

        # Group 5: Background & Current Work
        # Q: What’s your background, and what are you currently working on?
        # Q: Who are you and what are you currently working on?
        # Q: What's your backstory?
        if "background" in df.question[i] or "What's your backstory" in df.question[i] \
                or "Who are you and what are you currently working on?" in df.question[i]:
            df.at[i, "question"] = "background_and_current_work"

        # Group 6: Disadvantages
        # Q: What were your biggest disadvantages (different wordings)
        if "disadvantages" in df.question[i]:
            df.at[i, "question"] = "disadvantages"

        # Group 7: Revenue & Losses
        # Q: In the end, how much money did you lose?
        # Q: Which were your revenues, expenses and/or losses?
        if "lose" in df.question[i] or "expense" in df.question[i]:
            df.at[i, "question"] = "revenue_expenses_losses"

        # Group 8: Cause of Failure
        # Q: What were the causes of failure?
        # Q: When did things start to go in the wrong direction?
        # Q: Which was the problem with your product/service? How did you realize?
        # Q: What did eventually cause the shut-down?
        # Q: What made you want to exit MealSurfers?
        if "wrong direction" in df.question[i] or "failure" in df.question[i] or "shut down" in df.question[i] \
                or "Which was the problem with your product/service? How did you realize?" in df.question[i] \
                or "What made you want to exit" in df.question[i] \
                or "What did make you finally shut" in df.question[i]:
            df.at[i, "question"] = "causes_of_failure"

        # Group 9: Main Lessons & Advice
        # Q: Since starting your startup, what have been your main lessons?
        # Q: What did you learn and what's your advice for young entrepreneurs?
        # Q: If you had the chance to do things differently, what would you change?
        if "advice" in df.question[i] or "main lessons" in df.question[i] \
                or "do things differently" in df.question[i]:
            df.at[i, "question"] = "main_lessons_and_advice"

        # Group 10: From Idea to Product
        # Q: How did you build it?
        # Q: How did you go from idea to product?
        # Q: Which is your business model?
        # Q: Which was your business plan?
        if "How did you build it?" in df.question[i] \
                or "How did you go from idea to product?" in df.question[i] \
                or "business model" in df.question[i] or "business plan" in df.question[i]:
            df.at[i, "question"] = "from_idea_to_product"

        # Group 11: Growth of Startup & Customers
        # Q: How did you grow it?
        # Q: What were your key levers to start growing your startup?
        # Q: Since launch, what has worked to attract and retain customers?
        # Q: How did Lockpick Entertainment get their first customers?
        if "How did you grow it?" in df.question[i] or "key levers" in df.question[i] \
                or "to attract and retain customers" in df.question[i] \
                or "first customers" in df.question[i]:
            df.at[i, "question"] = "growth_of_startup_and_customers"

        # Group 12: Current State & Goals for Future
        # Q: How are you doing today and what are your goals for the future?
        # Q: What is the current state of the project?
        if "current state" in df.question[i] or "goals for the future" in df.question[i]:
            df.at[i, "question"] = "current_state_goals_future"

        # Group 13: Competition
        # Q: How were you different and better than your competition?
        # Q: With so much competition, how did you differ from them?
        if "competition" in df.question[i] or "compete" in df.question[i]:
            df.at[i, "question"] = "competition"

        # Group 14: More infos about startup
        # Q: Where can we go to learn more?
        if "Where can we go to learn more?" in df.question[i]:
            df.at[i, "question"] = "more_infos_about_startup"

    # STEP 4.3: Merge and transform into columns (long to wide)
    df_transformed = df.pivot_table(index=["startup"], columns='question', values='answer',
                                    aggfunc=lambda x: ' '.join(x)).reset_index()

    # STEP 4.4: Drop questions that only have been asked to one interviewee and could not be grouped
    # Q: How do you find time to work on so many projects at the same time?
    # Q: How was it like being part of Microsoft Azure Accelerator powered by Techstars?
    # Q: How was it like to participate in Shark Tank?
    # Q: As a B2B2C business, how did you validate both sides?
    # Q: How did the launch go?
    df_transformed = df_transformed.drop(df_transformed.columns[[1, 2, 3, 4, 5]], axis=1)

    # STEP 5: Merge dataframes by startup
    df_general = pd.DataFrame.from_dict(startup_data_general)
    final_data = pd.merge(df_transformed, df_general, on='startup', how="left")

    # STEP 6: Export to CSV
    final_data.to_csv('../dat/startup_data_final.csv', header=True)

if __name__ == '__main__':
    app.run(main)
