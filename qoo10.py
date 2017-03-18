# -*- coding: utf-8 -*-
import MySQLdb
import time
import sys
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver


class create_table:
    def __init__(self):
        self.db = MySQLdb.connect(user='root',passwd='Wwj@200859',db='qoo10',charset="utf8", host='127.0.0.1')

    def crate(self):
        cursor = self.db.cursor()

        # create mysql table
        sql_commodity = """CREATE TABLE IF NOT EXISTS `commodity`(
                            `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'incrateID',
                            `text` text NOT NULL COMMENT 'commodity text',
                            `shipping_fee` varchar(255) NOT NULL COMMENT 'answer',
                            `price` varchar(255) NOT NULL COMMENT 'price',
                            `discount` varchar(255) COMMENT 'price after discount',
                            `shop_name` varchar(255) COMMENT 'shop_name',
                            `shop_link` varchar(255) COMMENT 'shop_link',
                            `shop_rate` varchar(255) COMMENT 'shop_rate',
                            `image` varchar(255) COMMENT 'commodity image',
                            `commodity_link` varchar(255) COMMENT 'commodity_link',
                            PRIMARY KEY (`id`)
                        ) ENGINE=InnoDB  ;"""


        sql_comment = """CREATE TABLE IF NOT EXISTS `comment` (
                        `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'commentID',
                        `text` text NOT NULL COMMENT 'comment text',
                        `commodity_id` int(18) NOT NULL COMMENT 'commodityID',
                        `writer` varchar(255) NOT NULL COMMENT 'writer',
                        `date` varchar(255) NOT NULL COMMENT 'write_time',
                        `rating` varchar(255) NOT NULL COMMENT 'recommend',
                        PRIMARY KEY (`id`)
                    ) ENGINE=InnoDB ;"""

        cursor.execute(sql_commodity)
        cursor.execute(sql_comment)
        self.db.close()


# data clean class
class Tool:

    removeADLink = re.compile('<div class="link_layer.*?</div>')
    removeImg = re.compile('<img.*?>| {1,7}|&nbsp;')
    removeAddr = re.compile('<a.*?>|</a>')
    replaceLine = re.compile('<tr>|<div>|</div>|</p>')
    replaceTD = re.compile('<td>')
    replaceBR = re.compile('<br><br>|<br>')
    removeExtraTag = re.compile('<.*?>')
    removeNoneLine = re.compile('\n+')

    def replace(self, x):
        x = re.sub(self.removeADLink, "", x)
        x = re.sub(self.removeImg, "", x)
        x = re.sub(self.removeAddr, "", x)
        x = re.sub(self.replaceLine, "\n", x)
        x = re.sub(self.replaceTD, "\t", x)
        x = re.sub(self.replaceBR, "\n", x)
        x = re.sub(self.removeExtraTag, "", x)
        x = re.sub(self.removeNoneLine, "\n", x)
        return x.strip()  # get the answers of question


class Mysql:
    # get current time
    def getCurrentTime(self):
        return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))

    # initialize database
    def __init__(self):
        try:
            self.db = MySQLdb.connect(user='root',passwd='Wwj@200859',db='qoo10',charset="utf8", host='127.0.0.1')

            # get cursor()
            self.cur = self.db.cursor()
        except MySQLdb.Error,e:
            print self.getCurrentTime(),'connect database false，reason%d: %s' % (e.args[0], e.args[1])

    # 插入数据
    def insertData(self,table,my_dict):
        try:
            cols = ','.join(my_dict.keys())
            values = '","'.join(my_dict.values())
            sql = 'INSERT INTO %s (%s) VALUES (%s)' % (table, cols, '"' + values + '"')
            try:
                result = self.cur.execute(sql)
                # get insert line id
                insert_id = self.db.insert_id()
                # submit to database excute
                self.db.commit()
                # judge if succeed
                if result:
                    return insert_id
                else:
                    return 0

            except MySQLdb.Error,e:
                # if false rollback
                self.db.rollback()
                # the key unique, can`t insert
                if "key 'PRIMARY'" in e.args[1]:
                    print self.getCurrentTime(), "data exist, false "
                else:
                    print self.getCurrentTime(), "insert data false，reason %d: %s" % (e.args[0], e.args[1])

        except MySQLdb.Error, e:
            print self.getCurrentTime(), "database error ，reason%d: %s" % (e.args[0], e.args[1])


# get review class
class get_comment:
    def __init__(self):
        self.tool = Tool()
        self.header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
        reload(sys)
        sys.setdefaultencoding('utf-8')
        self.mysql = Mysql()
        self.page = 1

    def getCurrentTime(self):
        return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))


    def get_commodity(self,url,number):
        time.sleep(3)
        list_commodity = []
        while self.page <= number:
            link = url + str(self.page)
            page = requests.get(link, headers=self.header).content
            print 'is gaining'+str(self.page)+'page commodity'
            soup = BeautifulSoup(page, 'lxml')
            comm_all = soup.find(name='div',attrs={'id':'div_gallery_new'})
            commodity = comm_all.find_all(name='li')
            for co in commodity:
                # get commodity name
                commodity_name = co.find(name='a',attrs={'class','tt'}).get_text()

                # get commodity rate
                rate_text = co.find(name= 'div', attrs={'class','dtl'}).get_text()
                # 此处不能用match，不明白为什么
                rate = re.search(r'rate\:.(.*)', str(rate_text))
                rate = rate.group(1)

                # get fee
                fee_all = co.find(name='div', attrs={'class','prc'})
                discount_fee = fee_all.find(name='strong').get_text()
                discount = re.search(r'S\$(.*)', discount_fee)
                discount = discount.group(1)
                try:
                    initial = fee_all.find(name='del').get_text().encode('utf-8')
                    initial_fee = re.search('S\$(.*)', initial)
                    initial_fee = initial_fee.group(1)
                except:
                    initial_fee = 'None'

                # get postage
                postage_rate = co.find(name='div',attrs={'class','ship_area'})
                try:
                    postage = postage_rate.find(name='strong').get_text().encode('utf-8')
                    ship_fee = re.search('S\$(.*)', postage)
                    ship_fee = ship_fee.group(1)
                except:
                    ship_fee = 0


                # get shop name and shop link
                shop_info = co.find(name='div', attrs={'class','shop'})
                shop_content = re.search('<a href="(.*?)".*?title="(.*?)">', str(shop_info), re.S)
                shop_name = shop_content.group(1)
                shop_link = shop_content.group(2)

                # get image of comoodity image and comoodity link
                image_info = co.find(name='a', attrs={'class','thmb'})
                image_re = re.search('<a class=.*?href="(.*?)".*?target.*?<img.*?gd_src="(.*?)"', str(image_info), re.S)
                commodity_link = image_re.group(1)
                commodity_image = image_re.group(2)

                list_commodity.append([shop_name,rate,shop_link,commodity_name,discount,initial_fee,ship_fee,commodity_link,commodity_image])
            self.page +=1
        return list_commodity


    # Implement paging capabilities
    def get_review_co(self,commodities):
        cal =1
        list_insert = []
        for co in commodities:
            print 'crawling'+str(cal)+'page information'
            # get commodity and store in database

            commo_dict ={
                        'text':str(co[3]),
                        'shipping_fee':str(co[6]),
                        'price':str(co[4]),
                        'discount':co[5],
                        'shop_name':co[0],
                        'shop_link':co[2],
                        'image':co[8],
                        'commodity_link':co[7],
                        'shop_rate':(co[1])
                        }
            insert_id = self.mysql.insertData('commodity', commo_dict)
            list_insert.append(insert_id)
        cal +=1
        return list_insert

    def get_review(self,commodities,insert_id):
        list_link_comm = []
        for co in commodities:
            list_link_comm.append(co[7])
        zip(insert_id,list_link_comm)
        cal = 1
        for z,n in zip(insert_id,list_link_comm):
            print 'crawling '+str(cal)+'commodity review'
            # 这里不能用sel.driver,不知道为什么，用webdriver.Chrome()就可以
            driver = webdriver.Chrome()
            driver.get(n)
            driver.maximize_window()
            target = driver.find_element_by_id("questionAnswer")
            driver.execute_script("arguments[0].scrollIntoView();", target)
            time.sleep(5)  # 睡3秒让网页加载完再去读它的html代码，必须要让计算器读完网页才能进行下一步，不然报错

            while 1:
                soup = BeautifulSoup(driver.page_source, 'lxml')
                feedback = re.findall(
                        'tr id="feedback_.*?<span.*?<span.*?>(.*?)</span>.*?<div.*?<a href=.*?>(.*?)</a>.*?<td>(.*?)</td>.*?<td>(.*?)</td>',
                        str(soup), re.S)
                for n in feedback:
                    #get review detail
                    review_dict ={
                                'text':n[1],
                                'commodity_id':str(z),
                                'writer':n[3],
                                'date':n[2],
                                'rating':n[0]
                            }
                    if self.mysql.insertData("comment", review_dict):
                        print self.getCurrentTime(), "store succeed"
                    else:
                        print self.getCurrentTime(), "store false"
                try:
                    page = soup.find(name='div', attrs={'id': 'feedback_paging'})
                    cu_page_number = page.find(name='a', attrs={'class', 'current'}).get_text().encode('utf-8')

                    final = re.search('<a.*?class="next".*?value="(.*?)">', str(page), re.S)
                    final_number = final.group(1).encode('utf-8')

                    if cu_page_number != final_number:
                        driver.find_element_by_xpath("//div[@id='feedback_paging']/a[@title='next']").click()
                        time.sleep(5)
                    else:
                        print "finish crawl"
                        break
                except:
                    print "finish crawl"
                    break
            driver.quit()
            cal +=1

    def main(self,url,number):
        c = create_table()
        c.crate()
        print 'finish creating database'
        commodity = self.get_commodity(url,number)
        insert_id = self.get_review_co(commodity)
        self.get_review(commodity,insert_id)


a = get_comment()
a.main('http://list.qoo10.sg/gmkt.inc/Category/Default.aspx?gdlc_cd=100000001&gdmc_cd=200000001&gdsc_cd=&keywordArrLength=1&brand_keyword=&keyword_hist=&keyword=&within_keyword_auto_change=&image_search_rdo=U&attachFile=&search_image_url=&search_image_nm=&search_keyword=&is_img_search_yn=N&sortType=SORT_RANK_POINT&dispType=UIG5&flt_pri_idx=-1&filterDelivery=NNNNNANNNN&search_global_yn=&basis=&shipFromNation=&shipto=&brandnm=&SearchNationCode=&is_research_yn=Y&hid_keyword=&orgPriceMin=&orgPriceMax=&priceMin=&priceMax=&category_specific_kw_nos=&trad_way=&curPage=1&pageSize=120&partial=on&brandno=&paging_value=',1)