# data_spider.py
import csv
import os
import re
import time
import random
import requests
import urllib3
from bs4 import BeautifulSoup

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ZhaopinSpider:
    def __init__(self, csv_filename='zhaopin_jobs.csv'):
        self.session = requests.Session()
        self.csv_filename = csv_filename
        # 更新请求头，模拟真实浏览器
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        # 禁用代理
        self.session.trust_env = False

        # 初始化CSV文件，写入表头
        self.init_csv()

    def init_csv(self):
        """初始化CSV文件，写入表头"""
        # 如果文件已存在，先删除它
        if os.path.exists(self.csv_filename):
            os.remove(self.csv_filename)

        # 重新创建文件并写入表头
        with open(self.csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['职位名称', '公司名称', '工作地点', '薪资', '工作经验', '学历要求'])

    def get_jobs(self, page=1, keyword='python'):
        # 使用更稳定的URL格式
        url = 'https://sou.zhaopin.com/'
        params = {
            'jl': '',
            'kw': keyword,
            'p': page,
        }

        try:
            # 禁用SSL验证，设置超时时间
            response = self.session.get(
                url,
                params=params,
                headers=self.headers,
                timeout=15,
                verify=False  # 禁用SSL验证
            )

            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            jobs = []

            # 更新选择器 - 根据智联招聘实际页面结构调整
            job_elements = soup.find_all('div', class_=re.compile(r'joblist|position'))

            if not job_elements:
                # 如果没有找到工作列表，尝试其他选择器
                job_elements = soup.find_all('li', class_=re.compile(r'job|item'))

            for job in job_elements:
                try:
                    # 职位基本信息
                    title_elem = job.find('a', class_=re.compile(r'jobinfo__name'))
                    company_elem = job.find('a', class_=re.compile(r'companyinfo__name'))

                    # 薪资信息
                    salary_elem = job.find('p', class_='jobinfo__salary')

                    # 工作地点
                    location_container = job.find('div', class_='jobinfo__other-info')
                    location_text = '未知地点'
                    experience_text = '经验不限'
                    education_text = '学历不限'

                    if location_container:
                        info_items = location_container.find_all('div', class_='jobinfo__other-info-item')
                        if len(info_items) >= 1:
                            # 第一个通常是地点信息
                            location_span = info_items[0].find('span')
                            if location_span:
                                location_text = location_span.get_text().strip()

                        if len(info_items) >= 2:
                            # 第二个通常是工作经验
                            experience_text = info_items[1].get_text().strip()

                        if len(info_items) >= 3:
                            # 第三个通常是学历要求
                            education_text = info_items[2].get_text().strip()

                    if all([title_elem, company_elem]):
                        job_data = {
                            'title': title_elem.get_text().strip(),
                            'company': company_elem.get_text().strip(),
                            'location': location_text,
                            'salary': salary_elem.get_text().strip() if salary_elem else '面议',
                            'experience': experience_text,
                            'education': education_text
                        }
                        jobs.append(job_data)

                except Exception as e:
                    continue

            return jobs

        except Exception as e:
            print(f"获取第{page}页数据失败: {str(e)}")
            return []

    def save_to_csv(self, jobs):
        """将数据保存到CSV文件"""
        if not jobs:
            return

        with open(self.csv_filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            for job in jobs:
                # 写入一行数据
                writer.writerow([
                    job['title'][:200],      # 限制长度
                    job['company'][:200],
                    job['location'][:100],
                    job['salary'][:100],
                    job['experience'][:50],
                    job['education'][:50]
                ])

        print(f"成功保存 {len(jobs)} 条数据到CSV文件")

    def run(self, keywords=['python', 'java'], max_pages=50):
        print("开始爬取智联招聘岗位数据...")
        all_jobs = []

        for keyword in keywords:
            print(f"正在搜索关键词: {keyword}")
            for page in range(1, max_pages + 1):
                print(f"正在爬取第{page}页...")
                jobs = self.get_jobs(page, keyword)
                if jobs:
                    all_jobs.extend(jobs)
                    self.save_to_csv(jobs)
                    print(f"关键词'{keyword}'第{page}页获取到 {len(jobs)} 条数据")
                else:
                    print(f"关键词'{keyword}'第{page}页未获取到数据")

                # 随机延时，避免被封
                time.sleep(3 + random.random() * 2)

        print(f"爬取完成，共获取{len(all_jobs)}条数据")
        print(f"数据已保存到: {self.csv_filename}")


def main():
    """主函数"""
    spider = ZhaopinSpider('zhaopin_jobs.csv')
    keywords = ['python', 'java', 'go', 'c++', 'javascript']
    spider.run(keywords,50)

if __name__ == '__main__':
    main()
