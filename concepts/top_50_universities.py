import pandas as pd

if __name__ == "__main__":
    top_university_file = "TIMES_WorldUniversityRankings_2024.csv"

    df = pd.read_csv(top_university_file)

    top_university_names = df['name'].tolist()[:50]

    university_names_str = ', '.join(top_university_names)

    # TODO: Check top_50_universities.txt, remove special characters, and convert abbreviations to full names for uniqueness and consistency (e.g., UCL -> University College London)
    with open("top_50_universities.txt", "w", encoding='utf-8') as file:
        for university in top_university_names:
            file.write(university + "\n")
