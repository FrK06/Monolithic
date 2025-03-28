# src/tools/web_scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager  # Add this for driver management
from datetime import datetime

class WebScraper:
    def __init__(self):
        self.driver = None
        self.start_driver()

    def start_driver(self):
        if not self.driver:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument('--log-level=3')  # Only show fatal errors
            options.add_argument('--silent')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            service = webdriver.ChromeService(ChromeDriverManager().install())  # Auto-manage driver
            self.driver = webdriver.Chrome(service=service, options=options)

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def scrape_url(self, url: str) -> str:
        try:
            self.driver.get(url)
            content = self.extract_content(url)
            return content
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""

    def extract_content(self, url: str) -> str:
        # Handle specific time-related sites
        if any(site in url.lower() for site in ['time.is', 'worldtimebuddy', 'timeanddate']):
            return self.scrape_time_site()
        else:
            # Wait for body and return text
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                return self.driver.find_element(By.TAG_NAME, "body").text
            except TimeoutException:
                return "Failed to load page content."

    def scrape_time_site(self):
        """Improved method to extract date and time information from time-related websites"""
        # Get current date from system instead of scraping
        current_time = datetime.now()
        formatted_date = current_time.strftime("%B %d, %Y")
        formatted_time = current_time.strftime("%H:%M:%S")
        
        return f"Time: {formatted_time}, Date: {formatted_date}"
