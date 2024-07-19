import pandas as pd

if __name__ == "__main__":
    top_university_file = "TIMES_WorldUniversityRankings_2024.csv"

    df = pd.read_csv(top_university_file)

    university_names = df['name'].tolist()[:500]

    for rank, name in enumerate(university_names, start=1):
        print(f"{rank}. {name}")
