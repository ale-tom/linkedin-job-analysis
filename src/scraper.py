import os
from dotenv import load_dotenv
import time
from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd

load_dotenv()  # Load environment variables from .env file


def login_to_linkedin(driver: webdriver.Chrome, username: str, password: str) -> None:
    """
    Log into LinkedIn using provided credentials.
    """
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    username_input = driver.find_element(By.ID, "username")
    password_input = driver.find_element(By.ID, "password")
    username_input.send_keys(username)
    password_input.send_keys(password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(3)


def extract_saved_job_urls(driver: webdriver.Chrome) -> List[str]:
    """
    Extract URLs of saved jobs from the LinkedIn saved jobs page.
    """
    saved_jobs_url = "https://www.linkedin.com/my-items/saved-jobs/"
    driver.get(saved_jobs_url)
    time.sleep(3)
    page_soup = BeautifulSoup(driver.page_source, "html.parser")

    job_links = []
    # TODO: This selector may need updating based on LinkedIn's current DOM structure
    for a_tag in page_soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/jobs/view/" in href:
            job_links.append(href)
    return list(set(job_links))  # Remove duplicates


def extract_job_requirements(driver: webdriver.Chrome, job_url: str) -> str:
    """
    Extract the job requirements text from a given job posting page.
    """
    driver.get(job_url)
    time.sleep(3)
    page_soup = BeautifulSoup(driver.page_source, "html.parser")

    # TODO: Adjust the selector below to target the specific element containing requirements
    requirements = ""
    requirements_section = page_soup.find("section", {"class": "description"})
    if requirements_section:
        requirements = requirements_section.get_text(separator="\n", strip=True)
    return requirements


def main():
    """
    Pipeline to extract job requirements from saved LinkedIn jobs and save the data to a CSV file.
    """
    username = os.environ.get("LINKEDIN_USERNAME")
    password = os.environ.get("LINKEDIN_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing LINKEDIN_USERNAME or LINKEDIN_PASSWORD environment variable."
        )

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        login_to_linkedin(driver, username, password)
        job_urls = extract_saved_job_urls(driver)

        data = []
        for url in job_urls:
            req_text = extract_job_requirements(driver, url)
            data.append({"job_url": url, "requirements": req_text})

        df = pd.DataFrame(data)
        df.to_csv("linkedin_job_requirements.csv", index=False)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
