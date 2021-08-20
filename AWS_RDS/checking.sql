# check if the system works well
USE twitter;
SELECT * FROM twitter.tweets;
SELECT * FROM twitter.tweets WHERE `date` = '2021-08-20' AND `hour`= 15;
SELECT * FROM twitter.tweets WHERE `tweet_id` = 1428737936216887296;
