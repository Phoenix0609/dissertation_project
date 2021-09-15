# check if the system works well
USE twitter;
SELECT * FROM twitter.tweets;
SELECT * FROM twitter.tweets WHERE `date` = '2021-08-20' AND `hour`= 15;
