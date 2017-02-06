library(readr)
library(dplyr)
library(ggplot2)
library(stringr)
library(rvest)
library(caret)
rounds = read_csv("rounds.csv")
companies = read_csv("companies.csv")
organizations = read_csv("organizations.csv")
rounds = mutate(rounds, year_funded = str_split(rounds$funded_at, "-", simplify = TRUE)[, 1])
organizations = rename(organizations, current_url = homepage_url)

# at first, just look at companies funded after 2012
d = rounds %>% filter(company_country_code %in% c("USA", "CAN") & funding_round_type == "seed" & 
                        year_funded > 2011)
d = d[!is.na(d$raised_amount_usd), ]
d = d %>% group_by(company_name) %>% summarize(raised_amount_usd = mean(raised_amount_usd))
d = left_join(d, companies %>% select(name, homepage_url), 
              by=c("company_name" = "name"))
d = left_join(d, organizations %>% select(name, short_description, current_url, facebook_url,
                                          twitter_url, linkedin_url), by=c("company_name" = "name"))
d = d[!is.na(d$short_description), ]
# need to exclude the top 1% because they will skew my data 
d = d[d$raised_amount_usd < quantile(d$raised_amount_usd, 0.99), ]


#remove duplicates 
d = d %>% group_by(company_name) %>% arrange(desc(raised_amount_usd)) %>% dplyr::slice(1) %>% ungroup()

# yeo-johnson transform to remove skew
d$raised_amount_usd = as.numeric(d$raised_amount_usd)
preprocessData = data.frame(d$raised_amount_usd)
preprocessParams = preProcess(preprocessData, method=c("YeoJohnson"))
print(preprocessParams)
transformed = predict(preprocessParams, preprocessData)
d$transformed_amount = transformed$d.raised_amount_usd
shapiro.test(d$transformed_amount[1:4999])
qqnorm(d$transformed_amount[1:4999]);qqline(d$transformed_amount[1:4999], col = 2)


# take a look at money raised, original and transformed
d %>% ggplot(aes(x=raised_amount_usd), fill = "lightblue") +
  geom_density(alpha = 0.5, fill = "lightblue") + xlab("Amount Raised") + ggtitle("Seed Funding")
d %>% ggplot(aes(x=transformed_amount), fill = "lightblue") +
  geom_density(alpha = 0.5, fill = "lightblue") + xlab("Amount Raised") + ggtitle("Seed Funding, Transformed")

# get some summary statistics for a few terms
terms = c("disrupt", "paradigm", "hack", "transform")
stat_table = d %>% mutate(term = "All Companies") %>% select(term, transformed_amount)
for (i in 1:length(terms)){
  data_temp = d[grepl(terms[i], d$short_description, ignore.case = TRUE),]
  data_temp$term = terms[i]
  data_temp = select(data_temp, term, transformed_amount)
  stat_table = bind_rows(stat_table, data_temp)
}
ggplot(stat_table, aes(x=term, y=transformed_amount)) + 
  geom_boxplot(outlier.shape=NA, color = "blue", fill = "lightblue") + 
  geom_jitter(position=position_jitter(width=.1, height=0), size = 0.1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size=14), 
        legend.position="bottom") + xlab("") + ylab("Amount Raised (transformed)") + 
  ggtitle("Buzzwords and startup funding")
ggsave("plot1.png")

p_values_1 = rep(NA, length(terms))
d_data = d$transformed_amount
for(i in 1:length(terms)){
  y1 = d[grepl(terms[i], d$short_description, ignore.case = TRUE), ]
  y2 = d[!grepl(terms[i], d$short_description, ignore.case = TRUE), ]
  p_value = wilcox.test(y1$transformed_amount, y2$transformed_amount, alternative = "two.sided", paired = FALSE)
  p_values_1[i] = p_value$p.value
  print(i)
}
p_values_1 = p.adjust(p_values_1, method = "hochberg", n = length(p_values_1))
results_table_1 = tibble(terms = terms, adj_p_values = p_values_1)
results_table_1 = results_table_1 %>% arrange(adj_p_values) %>% filter(adj_p_values < 0.05)


# make a full bag of words model
library(text2vec)
library(data.table)
dd = d
setDT(dd)
setkey(dd, company_name)
set.seed(2016L)
prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(dd$short_description, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = dd$company_name, 
                  progressbar = TRUE)
vocab = create_vocabulary(it_train)
rm(dd)
vocab_table = vocab[[1]]
# keep words that appear in less than 25% of documents and in at least 4 documents
vocab_table = vocab_table[vocab_table$doc_counts > (nrow(d) * 0.01) & vocab_table$doc_counts < (nrow(d) * 0.25), ]
terms = vocab_table$terms
grep("paradigm", terms)

# let's do a bunch of t tests then correct for fdr
p_values = rep(NA, length(terms))
d_data = d$transformed_amount
for(i in 1:length(terms)){
  y1 = d[grepl(terms[i], d$short_description, ignore.case = TRUE), ]
  y2 = d[!grepl(terms[i], d$short_description, ignore.case = TRUE), ]
  p_value = wilcox.test(y1$transformed_amount, y2$transformed_amount, alternative = "two.sided", paired = FALSE)
  p_values[i] = p_value$p.value
  print(i)
}
p_values = p.adjust(p_values, method = "hochberg", n = length(p_values))
results_table = tibble(terms = terms, adj_p_values = p_values)
results_table = results_table %>% arrange(adj_p_values) %>% filter(adj_p_values < 0.05)

# get summary statistics for terms that are above adjusted p value cutoff
terms = results_table$terms
stat_table = d %>% mutate(term = "All Companies") %>% select(term, transformed_amount)
for (i in 1:length(terms)){
  data_temp = d[grepl(terms[i], d$short_description, ignore.case = TRUE),]
  data_temp$term = terms[i]
  data_temp = select(data_temp, term, transformed_amount)
  stat_table = bind_rows(stat_table, data_temp)
}
ggplot(stat_table, aes(x=term, y=transformed_amount)) + 
  geom_boxplot(outlier.shape=NA, color = "blue", fill = "lightblue") + 
  geom_jitter(position=position_jitter(width=.1, height=0), size = 0.1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size=14), 
        legend.position="bottom") + xlab("") + ylab("Amount Raised (transformed)") + 
  ggtitle("Terms Associated With Funding")
ggsave("plot2.png")

