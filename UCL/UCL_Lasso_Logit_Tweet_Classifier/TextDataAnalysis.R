#Text Data Analysis

#Prep Work & Packages ----------
load("data/tweets.Rda")

library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(tidyverse)
library(glmnet)
library(ggplot2)

#Text Preparation ----------
tweetCorpus <- corpus(tweets$text, docvars = tweets)

#turn corpus into document-term matrix; make everything lower-case, remove numbers, punctuation and stopwords; TF-IDF weighting; remove airline names
dfm_tweets0 <- tweetCorpus %>%
  tokens(remove_numbers=T,
         remove_punct=T,
         include_docvars=T,
         remove_symbols = T) %>%
  tokens_remove(stopwords("en")) %>%
  dfm(tolower=T)%>%
  dfm_tfidf()

dim(dfm_tweets0) #13,066 unique words

#eliminate rare words
dfm_tweets1 <- dfm_tweets0 %>% dfm_trim(min_docfreq = 5)
dim(dfm_tweets1)

#eliminate other words
dfm_tweets <- dfm_tweets1 %>% dfm_remove(c("@united", "@usairways", "@americanair", "@southwestair", "@jetblue", "@virginamerica", "flight", "amp", "aa", "get", "now", "united", "jetblue", "us"))
dim(dfm_tweets)


#Word Usage Analysis ----------

#overview
textstat_frequency(dfm_tweets, force=T)[1:15] #most used words
table(tweets$sentiment)#general proportion of each sentiment
table(tweets$airline)#number of tweets for each airline
tapply(tweets$sentiment, tweets$airline, table)#proportion of each sentiment per airline


#1)words most associated with negative & positive sentiment
bysentiment <- textstat_frequency(dfm_tweets,25,groups=sentiment, force=T)
bysentiment$group[bysentiment$group=="0"] <- "Positive"
bysentiment$group[bysentiment$group=="1"] <- "Negative"

ggplot(bysentiment[bysentiment$group=="Positive"], 
       aes(x=frequency,y=reorder(feature,frequency))) +
  facet_wrap(~group, scales="free") +
  geom_point() + 
  ylab("") +
  xlab("Frequency")+
  ggtitle("Words Most Associated with Positive Sentiment")

ggplot(bysentiment[bysentiment$group=="Negative"], 
       aes(x=frequency,y=reorder(feature,frequency))) +
  facet_wrap(~group, scales="free") +
  geom_point() + 
  ylab("") +
  xlab("Frequency")+
  ggtitle("Words Most Associated with Negative Sentiment")

#wordcloud
pos_tweets <- dfm_tweets[dfm_tweets$sentiment == "0",]
textplot_wordcloud(pos_tweets,
                   min_size = 0.5,
                   max_size=6,
                   color = "darkblue",
                   comparison=FALSE, 
                   max_words=50)

neg_tweets <- dfm_tweets[dfm_tweets$sentiment == "1",]
textplot_wordcloud(neg_tweets,
                   min_size = 0.5,
                   max_size=6,
                   color = "red",
                   comparison=FALSE, 
                   max_words=50)


#2)sentiment by airlines

#most-used words for each airline (positive & negative)
byairline_pos <- textstat_frequency(pos_tweets, 5, groups=airline, force=TRUE)
ggplot(byairline_pos, 
       aes(x=frequency,y=reorder(feature,frequency))) +
  facet_wrap(~group, scales="free") +
  geom_point() + 
  ylab("") +
  xlab("Frequency")

byairline_neg <- textstat_frequency(neg_tweets, 5, groups=airline, force=TRUE)
ggplot(byairline_neg, 
       aes(x=frequency,y=reorder(feature,frequency))) +
  facet_wrap(~group, scales="free") +
  geom_point() + 
  ylab("") +
  xlab("Frequency")


#Dictionary Classifier ------------
neg.words <- c("bad","worst", "terrible" , "cancelled","hold", "hours" ,"time", "delayed","gate", "phone") 
pos.words <- c("thanks", "thank", "great", "love", "awesome","much", "good", "best", "appreciate", "amazing")
dico <- dictionary(list(negative = neg.words, positive = pos.words))

#get sentiment score
sentiment_dico <- dfm_lookup(dfm_tweets,dictionary=dico)
sentiment_dico <- convert(sentiment_dico,to="data.frame")

#most negative
tweets$text[which.max(sentiment_dico$negative)]
#most positive
tweets$text[which.max(sentiment_dico$positive)]

#classify
sentiment_dico$score <- ifelse((sentiment_dico$positive - sentiment_dico$negative)>0,0,1)


#Lasso Logit Classifier --------
dfm_tweets_mod <- as.matrix(cbind(tweets$sentiment, dfm_tweets))

#cross-validation
cv.rows <- sample(nrow(dfm_tweets_mod),(nrow(dfm_tweets_mod)/2))
cv.data <- dfm_tweets_mod[cv.rows,]
test.data <- dfm_tweets_mod[-cv.rows,]

lasso.tweets <- cv.glmnet(x=cv.data[,2:ncol(dfm_tweets_mod)],y=cv.data[,1],
                          family="binomial",type.measure="class")

#classify test set
tweets.preds <- predict(lasso.tweets,test.data[,2:ncol(dfm_tweets_mod)],type='class')

#ten words that most predict negative and positive reviews
lasso.coeffs <- as.matrix(coef(lasso.tweets)[coef(lasso.tweets)[,1]!=0,])
lasso.coeffs <- as.matrix(lasso.coeffs[order(lasso.coeffs[,1],decreasing = T),])
#negative
lasso.coeffs[1:10,]
#positive
lasso.coeffs[(nrow(lasso.coeffs)-10):nrow(lasso.coeffs),]

#Compare the Classifiers ####

#test dictionary based classifier
table(sentiment_dico$score,tweets$sentiment)

#test lasso logit based classifier
tweets.preds <- predict(lasso.tweets,test.data[,2:ncol(dfm_tweets_mod)],type='class')
table(tweets.preds,test.data[,1])