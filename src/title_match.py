
import re
from collections import defaultdict
import csv
import unittest
import sys
import MySQLdb

import tree

# way to turn this into a binary decision:
# make the fallback states:
# turn is_one_and_only flag on
# turn is_first_common flag on 


def norm_issue_title(issue_title):
	return re.compile(r'\s+').sub(' ', re.compile('[-,]|\([^)]+\)|\.$').sub(' ', issue_title)).strip().rstrip('.').strip().upper()

def norm_issuer_name(issuer_name):
	return re.compile(r'\s+').sub(' ', re.compile(r'[-,]|\.$').sub(' ', issuer_name)).strip()


discrim = tree.discrim

tests = tree.feature_collection(

    # thought: if this doesnt work so well, try chaining discriminators together
    
    discrim('icom', '(^|[^a-z])com(mon)?', field='reported_issue_title'),
    discrim('scom', '(^|[^a-z])com(mon)?', field='domain_security'),
    discrim('iclb', '(^|[^a-z])(class|series) b', field='reported_issue_title'),
    discrim('sclb', '(^|[^a-z])CL(ass)? B', field='domain_security'),
    discrim('scla', '(^|[^a-z])CL(ass)? A', field='domain_security'),
    discrim('clc', '(^|[^a-z])cl(ass)? C', field='reported_issue_title'),

    discrim('wclass', '(class|series) [^abc]', field='reported_issue_title', should_match=('Class Z',), dont_match=('Class A',)),
    discrim('anyclass', '(class|series) [a-z]', field='reported_issue_title', should_match=('Class A', 'Series A', 'Class Z')),
    discrim('award', 'award', field='reported_issue_title'),
    discrim('option', 'option', field='reported_issue_title'),
    discrim('per', '%', field='domain_security'),
    discrim('syma', 'A$', field='domain_security'),
    discrim('close', 'closed end fund', field='domain_security'),
    discrim('phantom', '(warrants?|options?|Debentures?|phantom)', field='reported_issue_title'),
    discrim('no', '^ *no +(beneficial +)?(shares|securit[a-z]+)', field='reported_issue_title'),
    discrim('par', ' par ', field='reported_issue_title'),
    discrim('spfd', '[^a-z]pfd[^a-z]', field='domain_security'),
    discrim('ipfd', '(^|[^a-z])(pfd|preferred|pref)($|[^a-z])', field='reported_issue_title', should_match=('pref.', 'pfd.', 'pfd')),
    discrim('ordshs', 'ord shs', field='domain_security'),
    discrim('iordshs', 'ordinary shares', field='reported_issue_title'),

    discrim('ires', 'restricted', field='reported_issue_title'),
    discrim('sres', 'restricted', field='domain_security'),
    discrim('isbi', 'shares of beneficial interest', field='reported_issue_title'),
    discrim('ssbi', '(^|[^a-z])sbi($|[^a-z])', field='domain_security'),
    discrim('premium', 'premium', field='reported_issue_title'),

    discrim('deferred', 'defer', field='reported_issue_title'),
    discrim('none', '^ *none *$', field='reported_issue_title', should_match=('none', ' none '), dont_match=('not none',)),

    discrim('prim', 'primary', field='domain_security'),
    discrim('ilim', 'limited', field='reported_issue_title'),

    discrim('dsub', '\.[a-z]$', field='domain_ticker_symbol', should_match=('BRK.A',), dont_match=('BRKA',)),
    discrim('rsub', '\.[a-z]$', field='reported_ticker_symbol', should_match=('BRK.A',), dont_match=('BRKA',)),
    discrim('dticklen', '.....', field='domain_ticker_symbol', should_match=('BLRKA',), dont_match=('BRK',)), 
    discrim('rticklen', '.....', field='reported_ticker_symbol', should_match=('BLRKA',), dont_match=('BRK',)), 
    discrim('dticka', '..a$', field='domain_ticker_symbol', should_match=('BRKA',), dont_match=('A', 'BBB')), 
    discrim('rticka', '..a$', field='reported_ticker_symbol', should_match=('BRKA',), dont_match=('A', 'BBB')), 
    discrim('dsub2', '[a-z][^a-z][a-z]$', field='domain_ticker_symbol', should_match=('BRK.A',), dont_match=('BRKA','^A')),
    discrim('rsub2', '[a-z][^a-z][a-z]$', field='reported_ticker_symbol', should_match=('BRK.A',), dont_match=('BRKA','^A')),

    discrim('nasdr', 'nasd', field='reported_exchange'),
    discrim('nasdd', 'nasd', field='domain_exchange'),
    discrim('sheetsr', 'sheets', field='reported_exchange'),
    discrim('sheetsd', 'sheets', field='domain_exchange'),

# can do len() like '.{32}' will match everything 32 or longer
)


db = MySQLdb.connect(db='financial_service')
if len(sys.argv) > 1 and sys.argv[1] == '-test':
    unittest.TextTestRunner(verbosity=1).run(tests.get_test_suite(db))
    raise SystemExit


domain = csv.reader(open('tickerData_EOL_domainList.txt'), delimiter='\t')

d = defaultdict(dict)

headers = domain.next()

for (company_name, security_description, ticker_symbol, exchange) in domain:
	d[company_name][security_description] = (ticker_symbol, exchange)


samplereader = csv.reader(open('new.csv'))

headers = samplereader.next()

rows = []

total = 0
total_skipped = 0

for (issuer_name, accession_number, issuer_id, issue_title, ticker_symbol, exchange) in samplereader:
	total += 1
	if issuer_name in d:
		found = False

		for security, (ticker_symbol1, exchange1) in d[issuer_name].iteritems():

			r = {}

			for test_name, test_obj in tests.iteritems():
				r[test_name] = test_obj.evaluate({'domain_security': security, 'reported_issue_title': issue_title, 'domain_ticker_symbol': ticker_symbol1, 'reported_ticker_symbol': ticker_symbol, 'reported_exchange': exchange, 'domain_exchange': exchange1})
				

			#r['len'] = len(d[issuer_name])

			truth = (d[issuer_name][security] == (ticker_symbol, exchange))

			r['__CLASS__'] = truth
			r['__DATA__'] = (issuer_name, d[issuer_name], security, (issue_title, ticker_symbol, exchange))

			rows.append(r)


	else:
		total_skipped += 1


t = tree.buildtree(rows, tree.giniimpurity)

t.printme()


t.prune(.1)

t.printme()

print t.sig_tests()

(total_e, total_t) =  t.expected_errors()
print total_e, total_t

print total_skipped
print total

print (total_t + total_skipped) / (total_t + total_e + total_skipped + 0.0)



t.print_mysql_classifier('title_matching', tests)
t.save_mysql_classifier(db, 'issue_title_match', tests)
