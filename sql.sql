【NA下跌原因用户调研】
gdw_na_ui_visit_log
11793264
21244911
dim_mobile_uid_cuid_map
145311863
nuomi_consume_prefer
808005815
===================================================
select cuid,passport_uid from dim_mobile_uid_cuid_map 
passid  string  passid
gender  string  性别：编码|编码解释|准确率
age     string  年龄：编码|编码解释|准确率
zodiac  string  星座：编码|编码解释|准确率
education       string  学历：编码|编码解释|准确率
period  string  人生阶段：编码|编码解释|准确率
industry        string  所在行业：编码|编码解释|准确率
profession      string  职业：编码|编码解释|准确率
assets_info     string  资产状况：编码|编码解释|准确率
con_level       string  消费水平：编码|编码解释|准确率
locate  string  常驻城市
marriage        string  婚姻状态：编码|编码解释|准确率

device_info     string  设备信息



trend   string  消费倾向，321分别代表高中低
============================================================================
insert overwrite local directory "/root/t.txt"
select * from dim_mobile_uid_cuid_map where cuid='b6a653cda9183b76cfa13c256ce85fe137bbc6a7'
select t1.cuid cuidy,t2.cuid cuidn from 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170601 and dt<=20170630) t1
left outer join 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170823 and dt<=20170903) t2
on t1.cuid=t2.cuid
===============================================================================================
生成手机号(不抽样)
insert overwrite local directory "/root/data_mobile_notchosen/"
row format delimited fields terminated by "\t"
select t4.cuid cuid,t5.passport_uid passport_uid,t5.mobile mobile
from 
(select t3.cuid_y cuid from 
(select t1.cuid cuid_y,t2.cuid cuid_n from 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170601 and dt<=20170630) t1
left outer join 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170823 and dt<=20170903) t2
on t1.cuid=t2.cuid) t3
where t3.cuid_n is null) t4
join (select cuid,passport_uid,mobile from dim_mobile_uid_cuid_map where cuid is not null) t5 on t4.cuid=t5.cuid;
         

==========================================================
询符合要求的用户画像(抽样)
(用户画像字段格式：编码|编码解释|准确率)
cuid	手机号	性别	年龄	学历	人生阶段	所在行业	资产状况	消费水平	常驻城市	婚姻状态
insert overwrite local directory "/root/data2/"
row format delimited fields terminated by "\t"
select /*+ mapjojn(t7)*/ 
t7.cuid cuid,t7.mobile mobile,
t8.gender gender,t8.age age,t8.education education,t8.period period,t8.industry industry,
t8.assets_info assets_info,t8.con_level con_level,t8.locate locate,t8.marriage marriage
from 
(select t6.cuid cuid,t6.passport_uid passport_uid,t6.mobile mobile from 
(select t4.cuid cuid,t5.passport_uid passport_uid,t5.mobile mobile
from 
(select t3.cuid_y cuid from 
(select t1.cuid cuid_y,t2.cuid cuid_n from 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170601 and dt<=20170630) t1
left outer join 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170823 and dt<=20170903) t2
on t1.cuid=t2.cuid) t3
where t3.cuid_n is null) t4
join (select cuid,passport_uid,mobile from dim_mobile_uid_cuid_map where cuid is not null) t5 on t4.cuid=t5.cuid) 
t6 distribute by rand() sort by rand() limit 2000) t7
join (select passid,gender,age,zodiac,education,period,industry,assets_info,con_level,locate,marriage from nuomi_consume_prefer ) t8 on t7.passport_uid=t8.passid


生成手机号(抽样)
insert overwrite local directory "/root/data_mobile/"
row format delimited fields terminated by "\t"
select t6.cuid cuid,t6.passport_uid passport_uid,t6.mobile mobile from 
(select t4.cuid cuid,t5.passport_uid passport_uid,t5.mobile mobile
from 
(select t3.cuid_y cuid from 
(select t1.cuid cuid_y,t2.cuid cuid_n from 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170601 and dt<=20170630) t1
left outer join 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170823 and dt<=20170903) t2
on t1.cuid=t2.cuid) t3
where t3.cuid_n is null) t4
join (select cuid,passport_uid,mobile from dim_mobile_uid_cuid_map where cuid is not null) t5 on t4.cuid=t5.cuid) t6
distribute by rand() sort by rand() limit 1000
===========================================================

insert overwrite local directory "/root/data_mobile2/"
row format delimited fields terminated by "\t"
select t6.cuid cuid,t6.passport_uid passport_uid,t6.mobile mobile from 
(select t4.cuid cuid,t5.passport_uid passport_uid,t5.mobile mobile
from 
(select t3.cuid_y cuid from 
(select t1.cuid cuid_y,t2.cuid cuid_n from 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170601 and dt<=20170630) t1
left outer join 
(select distinct cuid from gdw_na_ui_visit_log where dt>=20170823 and dt<=20170903) t2
on t1.cuid=t2.cuid) t3
where t3.cuid_n is null) t4
join (select cuid,passport_uid,mobile from dim_mobile_uid_cuid_map where cuid is not null) t5 on t4.cuid=t5.cuid) t6 
distribute by rand() sort by rand() limit 1000


=============================================================================
insert overwrite local directory "/root/data/"
row format delimited fields terminated by "\t"
select  
b.buy_month ,
count(distinct b.mobile) ,
count(b.order_id) ,
sum(b.total_money)/1000 as ReGMV
from  
(select  buy_month,uid,mobile,order_id,gift_card_money,discount_money,coupon_money,memcard_protect_fee,total_money
 FROM test2_gdw_summary_fugou_order_info where dt>=20170101 and dt<20170801 )b 
join
(select uid FROM test2_gdw_summary_fugou_order_info 
where dt>=20160101 and dt<20170101
group by uid)a 
on a.uid=b.uid
group by b.buy_month
order by b.buy_month
========================================================================================================
insert overwrite local directory "/root/20171227"
select t1.cuid,t1.idfa from 
(select cuid,max(idfa) as idfa from gdw_na_user_visit where dt=20171224 and  cuid is not null  group by cuid) t1
LEFT OUTER JOIN 
(select * from gdw_na_cuid_new where dt between 20161224 and 20171223) t2
on t1.cuid=t2.cuid
where t2.cuid is null;
===========================================================================================================

