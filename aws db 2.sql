show databases;
CREATE DATABASE twitter;
use twitter;
show tables;
DROP TABLE IF EXISTS tweets;

CREATE TABLE IF NOT EXISTS tweets (
`tweet_id` BIGINT(20) NOT NULL PRIMARY KEY,
`user_id` VARCHAR(25) DEFAULT NULL,
`location` text DEFAULT NULL,
`user_followers_count` int(10) unsigned DEFAULT NULL,
`date` date DEFAULT NULL,
`hour` TINYINT(2) DEFAULT NULL,
`tweet_text` text NOT NULL,
`text_len`	SMALLINT(3) UNSIGNED NOT NULL,
`top_1` VARCHAR(5) DEFAULT NULL,
`top_2`	VARCHAR(5) DEFAULT NULL,
`top_3` VARCHAR(5) DEFAULT NULL,
`polarity` TINYINT(1) DEFAULT NULL,
`subjectivity` TINYINT(2) DEFAULT NULL,

#`time` datetime DEFAULT NULL,
`is_retweet` BOOLEAN DEFAULT NULL,
`ori_tweet_id` VARCHAR(25) DEFAULT NULL,
`ori_user_id` VARCHAR(25) DEFAULT NULL,
`favorite_count` int(10) unsigned DEFAULT NULL,
`retweet_count` int(10) unsigned DEFAULT NULL,
`is_quoted` BOOLEAN DEFAULT NULL,
`quoted_id` VARCHAR(25) DEFAULT NULL,
`quoted_author_id` VARCHAR(25) DEFAULT NULL,
`is_reply` BOOLEAN DEFAULT NULL,
`reply_to_id` VARCHAR(25) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;














