library(ggplot2)
library(plyr)
library(caret)
library(reshape2)

temp <- tempfile()
download.file("http://www.football-data.co.uk/mmz4281/1415/data.zip",temp)
files <- lapply(unzip(temp),function(x) read.csv(x))
raw.data <- Reduce(function(x, y) merge(x, y, all=T,sort=F), files, accumulate=F)
unlink(temp)

data<-raw.data

h.odds <- data[c("Date","FTR","B365H","BWH","IWH","LBH","PSH","WHH","SJH","VCH","BbMxH","BbAvH")]
df <- na.omit(melt(h.odds,id.vars=c("FTR","Date"),variable.name="book",value.name="h.odds"))
wins <- data.frame(date=as.Date(df$Date,format="%d/%m/%y")
                   ,win=(df$FTR=="H"),book=df$book,pr=1/df$h.odds)
wins$pr=round(wins$pr,2)
x <- aggregate(win ~ pr, data=wins, mean)
x<-round(x,2)

df$Date <- as.Date(df$Date,format="%d/%m/%y")
backtest <- df['book'=="BbAvH",]
backtest <- backtest[with(backtest, order(Date)), ]
backtest <- backtest[backtest$h.odds >= 5 & backtest$h.odds <= 8.25 , ]
backtest$win <- backtest$FTR=="H"
backtest$returns <- backtest$win * backtest$h.odds - 1


#' Okay, so bookies give us odds on how a game might turn out. We've established that they are
#' usually quite correct in terms of estimating how likely the home team is to win:
p1 <- ggplot(data=x,aes(x=pr,y=win))+geom_abline(colour="red")+geom_point()+geom_line()+
  xlab("Implied Probability of Home Win (from bookie odds)") +
  ylab("Actual Outcome (aka posterior probability)") +
  ggtitle("How Right is the House")
plot(p1)

#' The red line is what things would look like if bookies were always right. 
#' In the graph above, anywhere that the points are not exactly on the red line is 
#' a potential _edge_: the bookies are wrong here.
#' 
#' However, since we are using data, we need to determine where we might have a systematic edge,
#' one that is not just showing up here due to random chance. The first step is to simplify what we have.
#' We've already done this a bit to get the previous graph, but now we'll remove all of these points
#' and replace them with a regression model, essentially a function.
p2 <- ggplot(data=x,aes(x=pr,y=win))+geom_abline(colour="red")+
  stat_smooth(method="lm",formula=y~poly(x, 3),size=1)+
  xlab("Implied Probability of Home Win (from bookie odds)") +
  ylab("Actual Outcome (aka posterior probability)") +
  ggtitle("How Right is the House")
plot(p2)

#' So we can see that even then the house doesn't always get things right and, in fact, there are some areas
#' where they are systematically wrong. Maybe we can exploit these, but to figure out how
#' we need to think about how gambling actually works.
#' 
#' The house tells us their expected outcome in the form of odds, which we can choose to bet against.
#' We've backed out the expected probability using a pretty straightforward transformation on the odds
#' (one which we've discussed: Implied Probability = 1/(Decimal Odds)), so we can reverse it to represent this function:

p3 <- ggplot(data=x,aes(x=pr,y=1/pr)) + geom_point() + geom_line() +
  xlab("Implied Probability of Home Win (from bookie odds)") +
  ylab("Decimal Odds (as they would be at the sportsbook)") +
  ggtitle("Odds are Probabiities are Odds")
plot(p3)

#' You may recognize this function. It's the top right quadrant of y=1/x.
#' 
#' Anyway, so now we have:
#' -The bookie's odds, which tell us the potential payout
#' -The actual outcome, which tells us how much we'd have made
#' So let's break down how much we would have made, on average, per bet at each of the different
#' levels the bookie could have given us. While we will use the decimal odds to calculate our payouts,
#' we'll keep our x-axis the same since odds can range widely (and probabilities cannot).
#' Read this graph as "If I bet $1 whenever the implied probability is X%, 
#' I would on average get $Y in profit. Here we go:
ret <- with(x,(win*1/pr-1))
p4 <- ggplot(data=x,aes(x=pr,y=ret,colour=ret>0)) + geom_point() +
  geom_hline(colour='red') + xlab("House Odds (as probabilities") +
  ylab("Returns per $1 bet") + ggtitle("Payouts by Odds")
plot(p4)

#' Even more apparent as a bar chart:
p4 <- ggplot(data=x,aes(x=pr,y=ret,colour=ret>0, fill=ret>0)) + geom_bar(stat="identity", ymin=-1)+ #geom_point()
  geom_hline(colour='red') + xlab("House Odds (as probabilities") +
  ylab("Returns per $1 bet") + ggtitle("Payouts by Odds")
plot(p4)

#' So, the conclusion is that if we are looking for places too seek an edge, these home team underdogs
#' might be a good place to start. The house seems to have some level of error when .
#' 
#' For shits and giggles, let's see what would have happened if we 
#' had bet entirely on home team underdogs in the range we are seeing as promising
#' (let's call it probabilities between 12%-20%, aka ~8.25-5). Part of our data is 
#' odds from an odds aggregator, so we'll use those for the simulation.

p5 <- ggplot(data=backtest,aes(x=Date,y=cumsum(returns)))+geom_line()+
  xlab("Date")+ylab("Cumulative Returns")+ggtitle("How Would We Have Done")
plot(p5)

#' So had we bet $1 when applicable starting in October, we would have been at a net gain of $50-60
#' by now. Not bad. And we haven't even thought about what game we're betting on yet!

#' There are some issues with this analysis, including that we just backtested our strategy
#' on the data we used to come up with the hypothesis (big no no!). But you get the point, and it's 
#' as good a place as any to start looking for an edge. 
