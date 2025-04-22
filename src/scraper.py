import os
import re
import json
from dotenv import load_dotenv
import time
from typing import List, Dict
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


# TODO: Extract the saved job URLs from different pages
def extract_saved_job_urls(driver: webdriver.Chrome) -> List[str]:
    """
    Extract URLs of saved jobs from the LinkedIn saved jobs page.
    """
    saved_jobs_url = "https://www.linkedin.com/my-items/saved-jobs/"
    driver.get(saved_jobs_url)
    time.sleep(3)
    page_soup = BeautifulSoup(driver.page_source, "html.parser")

    job_links = []
    # This selector relies on LinkedIn's current DOM structure - it may need to be updated if LinkedIn changes their
    # layout.
    for a_tag in page_soup.find_all("a", href=True):  # Find all anchor tags
        # Check if the href contains the job view path
        href = a_tag["href"]
        if "/jobs/view/" in href:
            job_links.append(href)
    return list(set(job_links))  # Remove duplicates


# TODO: Extract the job title and other details
def extract_job_requirements(
    driver: webdriver.Chrome, job_url: str
) -> Dict[str, List[str]]:
    """
    Navigate to the given job_url, wait for it to load, and extract any
    requirement sections (e.g. "Person Specification", "Requirements",
    "Qualifications", "Skills", "Preferred Qualifications") as a mapping
    from heading to list of bullet points.
    """
    # Navigate to the job URL
    driver.get(job_url)
    # Wait for the page to load and parse the HTML
    # This is an arbitrary value and may need to be adjusted based on the actual loading time of the page
    time.sleep(3)
    page_soup = BeautifulSoup(driver.page_source, "html.parser")

    sections: Dict[str, List[str]] = {}
    container = page_soup.find(
        "div", class_="job-details-about-the-job-module__description"
    )
    if not container:
        return sections

    heading_pattern = re.compile(
        r"(?:(?:Person Specification|.*Requirements.*|.*Qualifications.*|"
        r"Nice to Have|What You'll Need| A Plus If You Also Have|.*Advantageous.*|"
        r".*You to Have.*|Your Experience|.*Experience.*|.*Qualifications.*|",
        r".*Bonus.*|.*Looking for.*|.*Must Have.*|.*Bring.*|.*If You.*|.*You Are.*|"
        r".*You Should Have.*|About You|Your Experience)|.*Skills.*)",
        re.IGNORECASE,
    )

    for strong in container.find_all("strong"):
        heading = strong.get_text(strip=True)
        if heading_pattern.fullmatch(heading):
            # Try to find a <ul> immediately after the <strong> in the DOM
            ul = None
            sib = strong.next_sibling
            while sib:
                if getattr(sib, "name", None) == "ul":
                    ul = sib
                    break
                sib = sib.next_sibling

            # Fallback: look anywhere after the <strong> for the next <ul>
            if not ul:
                ul = strong.find_next("ul")

            if ul:
                items = [li.get_text(strip=True) for li in ul.find_all("li")]
                sections[heading] = items

    return sections


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
            sections = extract_job_requirements(driver, url)
            data.append(
                {
                    "job_url": url,
                    "requirements": json.dumps(sections, ensure_ascii=False),
                }
            )

        df = pd.DataFrame(data)
        df.to_csv("linkedin_job_requirements.csv", index=False)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
