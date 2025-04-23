import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

from scraper import (
    login_to_linkedin,
    extract_saved_job_urls,
    extract_job_requirements,
)


class DummyElement:
    def __init__(self):
        self.keys_sent = []
        self.clicked = False

    def send_keys(self, value: str) -> None:
        self.keys_sent.append(value)

    def click(self) -> None:
        self.clicked = True


class DummyLoginDriver:
    def __init__(self):
        self.actions = []
        # map (By, selector) to DummyElement
        self.elements = {
            (By.ID, "username"): DummyElement(),
            (By.ID, "password"): DummyElement(),
            (By.XPATH, "//button[@type='submit']"): DummyElement(),
        }

    def get(self, url: str) -> None:
        self.actions.append(("get", url))

    def find_element(self, by: str, selector: str):
        self.actions.append(("find_element", by, selector))
        key = (by, selector)
        if key in self.elements:
            return self.elements[key]
        raise NoSuchElementException()


def test_login_to_linkedin(monkeypatch):
    driver = DummyLoginDriver()
    # avoid actual sleeping
    monkeypatch.setattr(time, "sleep", lambda _: None)

    login_to_linkedin(driver, "user@example.com", "securepass")

    # Check navigation to login page
    assert ("get", "https://www.linkedin.com/login") in driver.actions

    # Check that username and password fields received the correct inputs
    username_elem = driver.elements[(By.ID, "username")]
    password_elem = driver.elements[(By.ID, "password")]
    assert username_elem.keys_sent == ["user@example.com"]
    assert password_elem.keys_sent == ["securepass"]

    # Check that the submit button was clicked
    submit_elem = driver.elements[(By.XPATH, "//button[@type='submit']")]
    assert submit_elem.clicked is True


def test_extract_saved_job_urls_single_page(monkeypatch):
    html = """
    <html>
      <body>
        <a href="/jobs/view/12345?trk=abc">Job 1</a>
        <a href="/jobs/view/67890">Job 2</a>
        <button aria-label="Next" disabled>Next</button>
      </body>
    </html>
    """

    class SinglePageDriver:
        def get(self, url: str) -> None:
            pass

        @property
        def page_source(self) -> str:
            return html

        def find_element(self, by: str, selector: str):
            # Next button is disabled, so raise as if not found
            raise NoSuchElementException()

    driver = SinglePageDriver()
    monkeypatch.setattr(time, "sleep", lambda _: None)

    urls = extract_saved_job_urls(driver)
    # should strip query params and dedupe
    assert set(urls) == {"/jobs/view/12345", "/jobs/view/67890"}


def test_extract_saved_job_urls_multiple_pages(monkeypatch):
    pages = [
        """
        <html>
          <body>
            <a href="/jobs/view/11111">Job A</a>
            <button aria-label="Next">Next</button>
          </body>
        </html>
        """,
        """
        <html>
          <body>
            <a href="/jobs/view/22222">Job B</a>
            <button aria-label="Next" disabled>Next</button>
          </body>
        </html>
        """,
    ]

    class MultiPageDriver:
        def __init__(self, pages):
            self.pages = pages
            self.index = 0

        def get(self, url: str) -> None:
            self.index = 0

        @property
        def page_source(self) -> str:
            return self.pages[self.index]

        def find_element(self, by: str, selector: str):
            # if there is a next page, return a fake clickable
            if (
                by == By.XPATH
                and selector.startswith("//button[@aria-label='Next'")
                and self.index < len(self.pages) - 1
            ):

                class NextButton:
                    def __init__(self, driver):
                        self.driver = driver

                    def click(self):
                        self.driver.index += 1

                return NextButton(self)
            raise NoSuchElementException()

    driver = MultiPageDriver(pages)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    urls = extract_saved_job_urls(driver)
    assert set(urls) == {"/jobs/view/11111", "/jobs/view/22222"}


def test_extract_job_requirements(monkeypatch):
    html = """
    <div class="job-details-jobs-unified-top-card__tertiary-description-container">
      <span class="tvm__text tvm__text--low-emphasis">Cambridge, England, United Kingdom</span>
      <span class="tvm__text tvm__text--low-emphasis">·</span>
      <span class="tvm__text tvm__text--low-emphasis">1 week ago</span>
      <span class="tvm__text tvm__text--low-emphasis">·</span>
      <span class="tvm__text tvm__text--low-emphasis">43 people clicked apply</span>
    </div>
    <div class="job-details-about-the-job-module__description">
      <strong>Person Specification</strong>
      <ul>
        <li>Requirement one.</li>
        <li>Requirement two.</li>
      </ul>
      <strong>General Skills</strong>
      <ul>
        <li>Skill A</li>
      </ul>
      <strong>Unrelated Heading</strong>
      <ul>
        <li>Should not be captured.</li>
      </ul>
    </div>
    """

    class ReqDriver:
        def __init__(self, html):
            self._html = html
            self.requested = []

        def get(self, url: str) -> None:
            self.requested.append(url)

        @property
        def page_source(self) -> str:
            return self._html

    driver = ReqDriver(html)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = extract_job_requirements(driver, "https://job.url/1")

    # ensure driver.get was called correctly
    assert result  # non-empty dict
    assert driver.requested == ["https://job.url/1"]

    # Top-card fields
    assert result["location"] == "Cambridge, England, United Kingdom"
    assert result["posted"] == "1 week ago"
    assert result["num_applicants"] == "43 people clicked apply"

    # Requirements mapping
    reqs = result["requirements"]
    assert "Person Specification" in reqs
    assert "General Skills" in reqs
    assert "Unrelated Heading" not in reqs

    assert reqs["Person Specification"] == ["Requirement one.", "Requirement two."]
    assert reqs["General Skills"] == ["Skill A"]
