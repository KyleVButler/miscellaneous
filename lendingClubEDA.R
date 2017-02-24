# This data set contains information on loans from Lending Club, and has ~70 different variables. 
# I know that this data set came from Kaggle, but I had never looked at it closely.
# I am not going to be the guy that copies the winning python script that uses tensorflow or something
# to build a model that can predict charge-off with an ROC of 0.999. I want to focus on exploratory 
# data analysis, hypothesis generation, and find something interesting that will be useful for
# business intelligence. What would a lender most care about? Lenders have two principal concerns: How
# can I make the most money for my invested capital, and how can I prevent charge-off. 

# I need two response variables, one for the return on invested capital, and another for the 
# loan status (paid or charged-off). The loan status is already in the data set ("loan_status"). 
# How can we measure loan profitability? With net annualized return or return on equity. 
# I need a few variables to calculate this:
# length of the loan period - "issue_d" is the month loan was funded and "last_pymnt_d" for the month
# that the last payment was received. Now I need a formula for annual yield. I will just use:
# APY = ( (principal + gain) / principal ) ^ (12/months) - 1
# now I need to calculate the principal and profit for each loan.
# Let's look at the relevant variables for a few loans...

# look at the data
library(dplyr)
library(ggplot2)
library(caret)
lc = readr::read_csv("loan.csv")
dim(lc)
# look at na values
na_by_category = lapply(lc, FUN = function(x) sum(is.na(x) / length(x))) 
# do any categories have over 50% na?
na_by_category[na_by_category > 0.5]
# what is the typical loan status
table(lc$loan_status)
# view the variables relevant for calculating APY for fully paid and charged off loans
lc %>% filter(loan_status == "Fully Paid") %>% 
       select(c(id, funded_amnt, funded_amnt_inv,
                out_prncp, out_prncp_inv, recoveries, total_pymnt,
                     total_pymnt_inv, total_rec_int, total_rec_late_fee, total_rec_prncp,
                     recoveries)) %>% slice(1:5)
lc %>% filter(loan_status == "Charged Off") %>% 
       select(c(id, funded_amnt, funded_amnt_inv,
                out_prncp, out_prncp_inv, recoveries, total_pymnt,
                total_pymnt_inv, total_rec_int, total_rec_late_fee, total_rec_prncp,
                recoveries)) %>% slice(1:5)
# The profit is simple to calculate and APY is (total_pymnt/funded_amnt) ^ (12/months) - 1
# let's see how to find out how many months the loan was active
lc %>% filter(loan_status == "Fully Paid") %>% 
       select(c(id, funded_amnt, issue_d, last_pymnt_d)) %>% slice(1:5)
########### remember to put these in kable for markdown file
lc %>% filter(loan_status == "Charged Off") %>% 
       select(c(id, funded_amnt, issue_d, last_pymnt_d)) %>% slice(1:5)
# let's use lubridate to find out how long each loan was active
library(lubridate)
# coerce to the date format
lc[] = lc %>% purrr::map_if(names(.) %in% c("issue_d", "last_pymnt_d"), ~ dmy(stringr::str_c("01-", .)))
# find the number of months each loan was active
# this will only work for completed loans so remove currents   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# may have to start earlier
lc = filter(lc, last_pymnt_d < "2015-08-02")
lc = mutate(lc, months_active = month(last_pymnt_d) - month(issue_d) + 
                     (12 * (year(last_pymnt_d) - year(issue_d))))
table(lc$loan_status)

# a few loans were open for zero months, lets set those to one mmonth
lc$months_active[lc$months_active == 0] = 1
# now calculate the APY
lc = mutate(lc, apy = ( (total_pymnt/funded_amnt) ^ (12/months_active) ) - 1  )
# remove any entries that do not have values for entries used to calculate the apy
lc = lc[lc %>% select(issue_d, last_pymnt_d, total_pymnt, funded_amnt, months_active, apy) %>% 
  complete.cases(), ]
max(lc$apy)
median(lc$apy)
lc %>% ggplot(aes(apy)) + geom_density(color = "blue", fill = "light blue") 
table(lc$loan_status)
# there are a few outliers, I will remove them - about the top 0.05%
table(lc$apy > 0.4)
lc = filter(lc, apy < 0.4)
lc %>% ggplot(aes(apy)) + geom_density(color = "blue", fill = "light blue") 

# i will also make a normalized apy variable
# function to standardize a variable
standardizer = function(x) {
  preprocessParams = preProcess(data.frame(x), method=c("center", "scale", "YeoJohnson"))
  transformed_data = predict(preprocessParams, data.frame(x))
  transformed_data[[1]]
}

lc$transformed_apy = standardizer(lc$apy)
lc %>% ggplot(aes(transformed_apy)) + geom_density(color = "blue", fill = "light blue") 


# let's find some interesting variables by looking through the data dictionary - i just want to look closely
# at maybe 10 variables, and build a model with those - not put every variable into model right now
# addr state looks fun
us = map_data("state")
lc$addr_state = tolower(state.name[match(lc$addr_state, state.abb)])
lc %>% group_by(addr_state) %>% summarize(avg_apy = mean(apy)) %>% 
  filter(addr_state %in% unique(us$region)) %>% ggplot(aes(fill=avg_apy, map_id=addr_state)) + 
    geom_map(map=us) + 
  expand_limits(x = us$long, y = us$lat) + ylab("") + xlab("")
# let's do some feature engineering
# it looks like "Moynihan's law of the Canadian border"
# I am going to convert each state into its distance from Canada (using latitude of center of population
# as a proxy)
pop_centers = readr::read_csv("pop_center.txt")
pop_centers$STNAME = tolower(pop_centers$STNAME)
# make alaska equal washington and hawaii equal florida so they are not outliers
pop_centers$LATITUDE[pop_centers$STNAME == "alaska"] = pop_centers$LATITUDE[pop_centers$STNAME == "washington"]
pop_centers$LATITUDE[pop_centers$STNAME == "hawaii"] = pop_centers$LATITUDE[pop_centers$STNAME == "florida"]
lc = left_join(lc, pop_centers %>% select(STNAME, LATITUDE), by = c("addr_state" = "STNAME"))

# i have another hunch, let's match each state with its NAEP score (using 4th grade math 2015)
naep = readr::read_csv("naep.csv")
naep$state = tolower(naep$state)
naep = rename(naep, pisa_score = average_score)
lc = left_join(lc, naep, by = c("addr_state" = "state"))


# let's see if any of my new variables can explain the apy - doesn't look like it
ggplot(aes(x = LATITUDE, y = transformed_apy), data = lc) + geom_hex() + stat_smooth(color="red")
ggplot(aes(x = pisa_score, y = transformed_apy), data = lc) + geom_hex() + stat_smooth(color="red")
summary(lm(transformed_apy ~ LATITUDE, data = lc))
summary(lm(transformed_apy ~ pisa_score, data = lc))

# some more variables to include
ggplot(aes(x = annual_inc, y = apy), data = lc) + geom_hex() + stat_smooth(color="red")
# lol look at the guy with 10 million annual income who didn't pay back any of his loan
# this is a great variable

# next one also looks good
ggplot(aes(x = collections_12_mths_ex_med, y = apy), data = lc) + geom_hex() + stat_summary(color="red")
str(lc$collections_12_mths_ex_med)
# same with the one below
ggplot(aes(x = delinq_2yrs, y = apy), data = lc) + geom_hex() + stat_smooth(color="red")
str(lc$delinq_2yrs)
# next is the "desc" variable... I could spend all weekend on this, so I will ignore it for now
# i will ignore joint accounts because they are rare

# this looks good
ggplot(aes(x = dti, y = apy), data = lc) + geom_hex() + stat_smooth(color="red")
str(lc$dti)

# what to do about this, it's a factor, looks like a great variable
ggplot(aes(x = emp_length, y = apy), data = lc) + geom_boxplot() + stat_smooth(color="red")
str(lc$emp_length)

# emp_title, the employment title provided by the borrower, let's mess around with this.
# I am going to make this into a new feature, a la Gregory Clark's work
lc$emp_title = trimws(tolower(lc$emp_title))
# let's figure out how common a title is, chief fiancial officer is a rare title, and 
# possibly someone who can pay back a loan
lc$emp_title[1:20] # it looks like most people put down the name of their employer, not the job title
job_lookup = as.data.frame(table(lc$emp_title))
job_lookup = rename(job_lookup, job_title_frequency = Freq)
lc = left_join(lc, job_lookup, by = c("emp_title" = "Var1"))
ggplot(aes(x = job_title_frequency, y = apy), data = lc) + geom_hex() + stat_smooth(color="red")
# cool, I'm going to keep the new variable