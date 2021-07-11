import os
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta
from tqdm import tqdm
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

class ReviewCollector():
    PARSER = "html.parser"
    SLEEP_TIME = 1
    SCROLL_TIME = 8
    SHOW_MORE_XPATH = '//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div/main/div/div[1]/div[2]/div[2]/div'

    MONTH_DICT = dict(map(lambda x: (x[1], x[0]), list(enumerate(["January", "February", "March", "April", "May", 
                                 "June", "July", "August", "September", "October", "November", "December"], 1))))
    def __init__(self, chrome_p, sv_path):
        self.chrome_p = chrome_p
        self.sv_path = Path(sv_path)
        
        
    def start_driver(self) -> None:
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko")
        self.driver = webdriver.Chrome(self.chrome_p, options=options)        
        
    def get_soup(self) -> BeautifulSoup:
        webpage = self.driver.page_source
        soup = BeautifulSoup(webpage, self.PARSER)
        sleep(self.SLEEP_TIME)
        return soup

    def get_link(self, url: str) -> None:
        self.driver.get(url)
        sleep(self.SLEEP_TIME)

    def exists_xpath(self, xpath: str) -> bool:
        try:
            self.driver.find_element(By.XPATH, xpath)
            return True
        except NoSuchElementException:
            return False

    def click_button(self, xpath: str) -> None:    
        self.driver.find_element(By.XPATH, xpath).click()
        sleep(self.SLEEP_TIME)

    def parse_date_to_number(self, date: str) -> str:
        month, day, year = date.split()
        day = int(day.strip(","))
        month = int(self.MONTH_DICT[month])
        return f"{year}-{month:02d}-{day:02d}"
    
    def scroll_down(self):
        start = dt.now()
        end = start + timedelta(seconds=self.SCROLL_TIME)
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(self.SLEEP_TIME)
            if dt.now() > end:
                break
        
    def get_data(self, n_reviews=1000, previous_length=None):
        # self.get_link(url)
        # soup = self.get_soup()
        # total_reviews = int(soup.find("div", attrs={"class": "dNLKff"}).text.strip(" total").replace(",", ""))
        data = [("name", "date", "rate", "text", "helpful")]
        # if n_reviews > total_reviews:
        #     raise ValueError(f"You Cannot craw lower than {total_reviews}(n_reviews={n_reviews})")
        # else:
        #     print(f"Total Reviews: {total_reviews}")
        pbar = tqdm(total=n_reviews)
        start_crawl = True

        soup = self.get_soup()
        infos = soup.find_all(name="div", attrs={"class": "xKpxId zc7KVe"})
        print(f'Previous Length = {previous_length}')
        while start_crawl:
            # scroll down            
            if len(infos) < previous_length:
                for i in range(10):
                    self.scroll_down()
                    pbar.set_description(f"scrolled {i+1}")

                    found_show_more_btn = self.exists_xpath(self.SHOW_MORE_XPATH)
                    if found_show_more_btn:
                        self.click_button(self.SHOW_MORE_XPATH)
                    else:
                        # no more data
                        start_crawl = False

            # start get data
            soup = self.get_soup()
            infos = soup.find_all(name="div", attrs={"class": "xKpxId zc7KVe"})[previous_length:]
            reviews = soup.find_all(name="div", attrs={"class": "UD7Dzf"})[previous_length:]
            if len(reviews) != 0:
                for info, review in zip(infos, reviews):
                    name = info.find(name="span", attrs={"class": "X43Kjb"}).text
                    date = self.parse_date_to_number(info.find(name="span", attrs={"class": "p2TkOb"}).text)
                    helpful = int(info.find(name="div", attrs={"class": "XlMhZe"}).find_all("div")[2].text)
                    rate = int(info.find(name="div", attrs={"class": "pf5lIe"}).select("div")[0].attrs["aria-label"].split()[1])
                    text = review.find_all("span")
                    if text[1].text == "":
                        text = text[0].text
                    else:
                        text = text[1].text
                    pbar.update()
                    if (n_reviews - len(data) + 1) == 0:
                        pbar.set_postfix_str(f"done {len(data)-1}")
                        start_crawl = False
                    else:
                        data.append((name, date, rate, text, helpful))
                        previous_length += 1

        return data, previous_length
    
    def main(self, url, n_reviews):
        self.get_link(url)
        per_n_r = n_reviews//10
        previous_length = 0
        for i, n_r in enumerate(range(per_n_r, n_reviews+1, per_n_r)):
            data, previous_length = self.get_data(n_reviews=per_n_r, previous_length=previous_length)
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(sv_path / f"reviews{i}.tsv", sep="\t", index=False, encoding="utf-8")

def main(chrome_p, sv_path, url, n_reviews):
    collector = ReviewCollector(chrome_p, sv_path)
    collector.start_driver()
    collector.main(url, n_reviews=n_reviews)

if __name__ == "__main__":
    chrome_p = "./chrome/chromedriver.exe"
    sv_path = Path("./data")
    url = "https://play.google.com/store/apps/details?id=us.zoom.videomeetings&hl=en_US&gl=US&showAllReviews=true"
    n_reviews = 50000

    main(chrome_p, sv_path, url, n_reviews)