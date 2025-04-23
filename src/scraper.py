import os
import re
import json
from dotenv import load_dotenv
import time
from typing import Any, List, Dict, Set
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup, Tag
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
    Open the LinkedIn saved jobs page, then walk through each page
    via the 'Next' button, collecting all '/jobs/view/' links.
    """
    saved_jobs_url = "https://www.linkedin.com/my-items/saved-jobs/"
    driver.get(saved_jobs_url)
    time.sleep(3)
    job_links: Set[str] = set()  # Set to avoid duplicates
    # Loop until there are no more pages
    while True:
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # collect any job‑view links on this page
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/jobs/view/" in href:
                # strip tracking/query params if present
                job_links.add(href.split("?")[0])

        # attempt to click the “Next” pagination button
        try:
            # This XPath selects the button with aria-label 'Next' that is not disabled
            next_btn = driver.find_element(
                By.XPATH, "//button[@aria-label='Next' and not(@disabled)]"
            )
            next_btn.click()  # Click the button to go to the next page
            time.sleep(3)
        except (NoSuchElementException, Exception):
            # no enabled Next button → we're done
            break
    return list(job_links)


def extract_job_requirements(driver: webdriver.Chrome, job_url: str) -> Dict[str, Any]:
    """
    Navigate to job_url, wait for load, then extract:
      • location: from the top-card tertiary description container
      • posted: e.g. "1 week ago"
      • num_applicants: e.g. "43 people clicked apply"
      • requirements: mapping of any requirement-style headings to their bullets
    """
    driver.get(job_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    info: Dict[str, Any] = {
        "location": "",
        "posted": "",
        "num_applicants": "",
        "requirements": {},
    }

    # 1) Top-card tertiary info
    top = soup.find(
        "div",
        class_="job-details-jobs-unified-top-card__tertiary-description-container",
    )
    if top:
        # collect each low-emphasis span, skipping the “·” separators
        parts = [
            span.get_text(strip=True)
            for span in top.find_all("span", class_="tvm__text tvm__text--low-emphasis")
            if span.get_text(strip=True) != "·"
        ]
        if len(parts) > 0:
            info["location"] = parts[0]
        if len(parts) > 1:
            info["posted"] = parts[1]
        if len(parts) > 2:
            info["num_applicants"] = parts[2]

    # 2) Requirements sections
    container = soup.find("div", class_="job-details-about-the-job-module__description")
    if container:
        heading_pattern = re.compile(
            r"(?:(?:Person Specification|.*Requirements.*|.*Qualifications.*|"
            r"Nice to Have|What You'll Need|A Plus If You Also Have|.*Advantageous.*|"
            r".*You to Have.*|Your Experience|.*Experience.*|.*Qualifications.*|"
            r".*Bonus.*|.*Looking for.*|.*Must Have.*|.*Bring.*|.*If You.*|.*You Are.*|"
            r".*You Should Have.*|About You|Your Experience)|.*Skills.*)",
            re.IGNORECASE,
        )
        sections: Dict[str, List[str]] = {}

        for strong in container.find_all("strong"):
            heading = strong.get_text(strip=True)
            if not heading_pattern.fullmatch(heading):
                continue

            # look for the very next <ul> in the DOM
            ul = None
            sib = strong.next_sibling
            while sib:
                if isinstance(sib, Tag) and sib.name == "ul":
                    ul = sib
                    break
                sib = sib.next_sibling

            # fallback: anywhere after the <strong>
            if not ul:
                ul = strong.find_next("ul")

            if ul:
                items = [li.get_text(strip=True) for li in ul.find_all("li")]
                sections[heading] = items

        info["requirements"] = sections

    return info


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

        rows = []
        for url in job_urls:
            info = extract_job_requirements(driver, url)
            rows.append(
                {
                    "job_url": url,
                    "posted": info["posted"],
                    "location": info["location"],
                    "num_applicants": info["num_applicants"],
                    "requirements": json.dumps(
                        info["requirements"], ensure_ascii=False
                    ),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv("linkedin_job_requirements.csv", index=False)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
