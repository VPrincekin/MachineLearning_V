#!/bin/bash

mysql -hcp01-sys-razz05hunbu2q-qa11.cp01.baidu.com -P3306 -uroot -pFlzx3qc --default-character-set=utf8 -e "use open_platform;show tables;update t_query_list set f_status=2 where f_eng
ine_type='HIVE' and f_status in (0,1) and ((select unix_timestamp()-unix_timestamp(f_start_exec_time))>10800);quit;"



update t_query_list set f_status=2 where f_engine_type='HIVE' and f_status in (0,1);