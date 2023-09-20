# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

import scrapy
from scrapy import Request
from string import ascii_lowercase

class StatsSpider(scrapy.Spider):
    name = "fighter_stats"

    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': 'all_fighter_stats.json'
    }

    def __init__ (self, *args, **kwargs):
        super(StatsSpider, self).__init__(*args, **kwargs)
        self.stats = ['height', 'weight', 'reach', 'stance', 'dob', 'sslpm', 'ssa', 'ssapm', 'ssd', 'td_avg_15', 'td_acc', 'td_def', 'sub_avg_15']
        self.start_urls = list(map(lambda x: f'http://www.ufcstats.com/statistics/fighters?char={x}&page=all', list(ascii_lowercase)))

    def parse (self, response):
        for fighter_url in response.xpath("//table[@class='b-statistics__table']/tbody/tr/td/a/@href").extract():
            yield Request(fighter_url, callback=self.parse_fighter_page)

    def parse_fighter_page (self, response):
        row = dict(zip(self.stats, list(filter(lambda x: x != '', map(lambda y: y.strip(), response.xpath('//ul["b-list__box-list"]/li/text()').getall())))))
        row['name'] = response.xpath('//span[@class="b-content__title-highlight"]/text()').extract()[0].strip()
        record = response.xpath('//span[@class="b-content__title-record"]/text()').extract()[0].split(':')[1].strip().split('-')
        row['wins'] = record[0]
        row['losses'] = record[1]
        row['draws'] = record[2]
        yield row
