### NOTES DE LECTURE VERIPOL

## Références
[19] J.T. Hancock, L.E. Curry, S. Goorha, M. Woodworth, On lying and being
lied to: A linguistic analysis of deception in computer-mediated communi-
cation, Discourse Processes 45 (2007) 1–23.

[21] B. Heredia, T.M. Khoshgoftaar, J.D. Prusa, M. Crawford, Cross-domain sen-
timent analysis: An empirical investigation, in: IEEE International Confer-
ence on Information Reuse and Integration, IRI’16, pp. 160–165.

[22] B. Heredia, T.M. Khoshgoftaar, J.D. Prusa, M. Crawford, An investigation of
ensemble techniques for detection of spam reviews, in: IEEE International
Conference on Machine Learning and Applications, ICMLA’16, pp. 127 - 133

[32] F. Li, M. Huang, Y. Yang, X. Zhu, Learning to identify review spam, in: In-
ternational Joint Conference on Artificial Intelligence, IJCAI’11, volume 22,
pp. 2488–2493

[40] M.L. Newman, J.W. Pennebaker, D.S. Berry, J.M. Richards, Lying words:
Predicting deception from linguistic styles, Personality and social psychol-
ogy bulletin 29 (2003) 665–675.

[43] M. Ott, Y. Choi, C. Cardie, J.T. Hancock, Finding deceptive opinion spam
by any stretch of the imagination, in: Annual Meeting of the Association
for Computational Linguistics: Human Language Technologies-Volume 1,
Association for Computational Linguistics, pp. 309–319.

[47] P. Rosso, L.C. Cagnina, Deception detection and opinion spam, in: A Prac-
tical Guide to Sentiment Analysis, Springer, 2017, pp. 155–171.

[48] V.L. Rubin, Y. Chen, N.J. Conroy, Deception detection for news: three types
of fakes, Proceedings of the Association for Information Science and Tech-
nology 52 (2015) 1–4.

[59] L. Zhou, Y. Shi, D. Zhang, A statistical language modeling approach to on-
line deception detection, IEEE Transactions on Knowledge and Data Engi-
neering 20 (2008) 1077–1081.

## Indices utiles à notre projet

In general, the more details provided in the report, the more likely it is to be honest. - Page 2

- page 24 -

5.2. Morphosyntax
When looking at the morphosyntactic characteristics of reports, it is observed
that truthful reports tend to have the following peculiarities: i) High frequency
of clitic gerund verbs (variable VCLIger), clitic infinitive verbs (variable VCLI-
inf), Clitics and personal pronouns (variable PPX) and Clitic personal pronouns
(variable PPC). ii) High frequency of Demonstrative pronouns (variable DM) and
Articles (variable ART). iii) Very high occurrences of the verb “to be” in infinitive
24
(variable VSinf) and the portmanteau word, “to”(variable PAL). The high occur-
rence in truthful reports of both clitic verbs and nouns (and also of the verb “to
be”) represents a high presence of descriptions of interactions between actors in
a narrative. More importantly it describes reflexive actions happening both to the
victim or the aggressor. This fact highly contrasts with the description of false
reports shown next, where common and not reflexive nouns and verbs are pre-
dominant. This means that truthful reports are more centered in the story telling
of the dynamic of the felony and the interactions of both the victim and the ag-
gressor where clitic verbs and nouns represent that the action falls on the subject
of the sentence. It is a closer and more personal way of narrating the incident.
Also, the high frequency of demonstrative pronouns, articles or the word “to” in
comparison with the absence thereof in false reports reflects that truthful reports
tend to be richer in descriptions.

On the contrary, false reports tend to present the following syntactic charac-
teristics. i) High frequency of subordinating conjunctions that introduce finite
clauses (variable CSUBF, i.e. “barely”). This characteristic reflects that false
reports tend to present sentences that reflect lack of information being “barely”
most of the times part of sentences in the line of “s/he could barely see” or “s/he
barely remembers.” Similarly, false reports also present high ratios of negations
(variable NEG) in comparison with the length of the document. Examples of sen-
tences found with this characteristic are: “s/he cannot report more data,” “s/he has
not suffered injuries,” “s/he could not see,” “s/he could not recognize” or “s/he
did not attend a medical facility.” Interestingly, as shown next, these very same
sentences but in positive are very representative of truthful reports. ii) High per-
centage of the variable Document Sentences, which is the inverse of the average
length of sentences. Meaning that false reports seem to be characterized by shorter
sentences. iii) Finally, and contrarily to truthful reports that seem to be delineated
by reflexive nouns and verbs indicating that the action is being focused on the
actor of the sentences, high ratios of common nouns (variable NC) are identified
as properties of false reports. While truthful reports are characterized by high fre-
quency of reflexive verbs and nouns reflecting a personal description of actions
happening to the victim, false reports are delineated by enumerations of common
nouns, focusing more on objects rather than the dynamic of the felony. This is also
confirmed by a higher ratio of Document Concepts in false reports, that is unique
concepts, meaning that false reports are more similar to lists of enumerated facts
or objects than true reports, that focus more on the description of a limited number
of concepts.


## Outils utilisés

The methodology used is a combination of Natural Language Processing (NLP)[26] algorithms, feature selection methods and Machine Learning (ML)[29] classification algorithms. - Page 4

Detection of false or deceptive texts [40, 19, 21, 59]. These works, that are mostly centered on the domain of spam detection and fake reviews
[43, 32, 22, 47], make use of traditional ML techniques such as Naive Bayes, Support Vector Machine (SVM), logistic regression or decision trees [8, 25 - Page 4

- page 13 -
Ridge Logistic Regression
Support Vector Machines
Random Forest
Naive Bayes

The performance of the methods, combined with the feature selection procedure described, has been evaluated using using Leave-One-Out Cross-Validation (LOOCV). Table 1 shows some statistics - Accuracy (95% Confidence Interval),
Sensitivity, Specificity, Precision, Recall, F1, AUC.
