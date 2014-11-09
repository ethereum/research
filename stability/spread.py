import copy, re
from sets import Set
# NOTE THAT ALL FUNCTIONS HERE WORK BY RETURNING A NEW ARRAY, THERE ARE NO IN-PLACE MODIFICATION METHODS
# ALSO KEEP IN MIND THAT COLUMNS AND ROW INDICES START FROM ZERO (ie. column A -> 0, E -> 4, Z -> 25, etc)
# Scans a CSV line, keeping track of separators and quotes
def scanline(ln,sp=','):
  arr = []
  inq = False
  ind = 0
  buff = ""
  while ind < len(ln):
    if ln[ind] == '"':
       inq = not inq
       ind += 1
    elif ln[ind] == '\\':
       buff += ln[ind+1]
       ind += 2
    elif ln[ind] == sp and not inq:
       arr.append(buff)
       buff = ""
       ind += 1
    else:
       buff += ln[ind]   
       ind += 1
  arr.append(buff)
  return arr
# Load a CSV into a 2D array, automatically filling in unevenly wide rows to make the array square
def load(f,sp=','):
  array = [scanline(x,sp) for x in open(f,'r').readlines()]
  maxlen = 0
  for i in range(len(array)):
    if len(array[i]) > maxlen: maxlen = len(array[i])
  for i in range(len(array)):
    if len(array[i]) < maxlen: array[i] += [''] * (maxlen - len(array[i]))
  return array
# Apply a Porter stemmer to every cell in a given range of cols in an array (calling stem with just a list and no cols argument stems _every_ cell)
# Example outputs: manage, management, manager, managing -> manag; pony, ponies -> poni; reincarnate, reincarnated, reincarnation -> reincarn
def stem(li,cols=0):
  if cols == 0: cols = range(len(li[0]))
  import porter
  pstemmer = porter.PorterStemmer()
  newlist = copy.deepcopy(li)
  for i in range(len(li)):
    for j in cols:
      string = str(li[i][j])
      for ch in "'"+'"+[]?!\n': string = string.replace(ch,'')
      words = string.split(' ')
      newlist[i][j] = ' '.join([pstemmer.stem(x.strip().lower(),0,len(x.strip())-1) for x in words])
  return newlist
# Is a string a number?
def isnumber(s):
    t = re.findall('^-?[0-9,]*\.[0-9,]*$',s)
    return len(t) > 0
            
# Declutters (removes special characters, numerifies numbers) every cell, rules same as those for stem(li,cols=0)
def declutter(li,cols=0):
  if cols == 0: cols = range(len(li[0]))
  newlist = copy.deepcopy(li)
  for i in range(len(li)):
    for j in cols:
      string = str(li[i][j])
      for ch in "'"+'"+[]?!\n': string = string.replace(ch,'')
      words = string.split(' ')
      newlist[i][j] = ' '.join([x.strip().lower() for x in words])
      if isnumber(newlist[i][j]):
        newlist[i][j] = float(newlist[i][j])
  return newlist
# Generate a list of individual words occurring in a given column in a given array; useful for generating source lists to do n-grams from
def wordlist(li,col):
  wlist = []
  for i in range(len(li)):
      words = li[i][col].split(' ')
      for w in words:
        if w not in wlist: wlist.append(w)
  return wlist
# Generates a list of phrases (complete cell entries)
def phraselist(li,col):
  wlist = []
  for i in range(len(li)):
     phrase= li[i][col]
     if phrase not in wlist: wlist.append(phrase)
  return wlist
# Retrieve just a few columns from a given array to make a smaller (narrower) array
def cols(li,cols):
  result = []
  for i in range(len(li)):
    newline = []
    for c in cols: 
      if c >= 0: newline.append(li[i][c])
      else: newline.append(1)
    result.append(newline)
  return result
# Combine two possibly unsorted arrays matching rows by heading in headingcol1 in li1 and headingcol2 in li2
# setting linclusive = True makes sure every row in li1 makes it into the output, same with rinclusive and li2
# Recommended to do some kind of sort after splice is done
def splice(li1,li2,headingcol1,headingcol2,linclusive=False,rinclusive=False):
  s1 = sorted(li1,key=lambda x:x[headingcol1],reverse=True)
  s2 = sorted(li2,key=lambda x:x[headingcol2],reverse=True)
  l1 = len(s1[0])
  l2 = len(s2[0])
  ind1 = 0
  ind2 = 0
  output = []
  while ind1 < len(s1) and ind2 < len(s2):
    if cmp(s2[ind2][headingcol2],s1[ind1][headingcol1]) == 1:
      if rinclusive: output.append([s2[ind2][headingcol2]] + [''] * (l1-1) + s2[ind2][:headingcol2] + s2[ind2][headingcol2 + 1:])
      ind2 += 1
    elif cmp(s2[ind2][headingcol2],s1[ind1][headingcol1]) == -1:
      if linclusive: output.append([s1[ind1][headingcol1]] + s1[ind1][:headingcol1] + s1[ind1][headingcol1 + 1:] + [''] * (l2-1))
      ind1 += 1
    else:
      output.append([s1[ind1][headingcol1]] + s1[ind1][:headingcol1] + s1[ind1][headingcol1 + 1:] + s2[ind2][:headingcol2] + s2[ind2][headingcol2 + 1:])
      ind1, ind2 = ind1 + 1, ind2 + 1
  while ind1 < len(s1) and linclusive:
    output.append([s1[ind1][headingcol1]] + s1[ind1][:headingcol1] + s1[ind1][headingcol1 + 1:] + [''] * l2)
    ind1 += 1
  while ind2 < len(s2) and rinclusive:
    output.append([s2[ind2][headingcol2]] + [''] * l1 + s2[ind2][:headingcol2] + s2[ind2][headingcol2 + 1:])
    ind2 += 1
  return output
# Creates a wordlist sorted according to function f taken of an array with the results in the addcols in order
# eg. sorted_wordlist with addcols = [2,4,6], row is 1 2 4 8 16 32 64, f=lambda x:x[2]+x[1]+1.01*x[0] returns sorting key 84.04
def sorted_wordlist(li,wcol,addcols,f=lambda x:x[1],rev=True):
  return [x[0] for x in sorted(onegrams(li,wcol,addcols),key=f,reverse=rev)]
# Utility function, used by twograms, threegrams and fourgrams
def compose(arg):
  return ' '.join(sorted(list(Set(arg))))
# Calculate a total sum for every desired column for different exact matches in wcol, column -1 is implied to be 1 for every row
# for example, consider the array
# dog        20   3
# dog house  15   28
# cat        25   31
# cat        10   7
# dog        40   0
# house      10   14
# Doing pivot(li,0,[1,-1]) gives you the list:
# dog        60   2
# dog house  15   1
# cat        35   2
# house      10   1
# wlist allows you to restrict the table to a given wordlist
def pivot(li, wcol, addcols,wlist=0,sortkey=lambda x:1):
  if wlist == 0: wlist = phraselist(li,wcol)
  result = {}
  for i in range(len(wlist)):
    result[wlist[i]] = [0] * len(addcols)
  for i in range(len(li)):
    nums = []
    for ac in addcols:
      if ac >= 0:
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1] * 0.01)
        else: num = float(num)
      else: num = 1
      nums.append(num)
    if li[i][wcol] in result: result[li[i][wcol]] = [pair[0] + pair[1] for pair in zip(result[li[i][wcol]],nums)]
  array = []
  for word in result.keys():
    array.append([word] + result[word])
  return sorted(array,key=sortkey,reverse=True)
# Similar to a pivot table but looks at individual keywords. The example list above will return with onegrams(li,0,[1,2]):
# dog        75   3
# cat        35   2
# house      25   2
def onegrams(li, wcol, addcols,wlist=0,sortkey=lambda x: 1):
  if wlist == 0: wlist = wordlist(li,wcol) 
  result = {}
  for i in range(len(wlist)):
    result[wlist[i]] = [0] * len(addcols)
  for i in range(len(li)):
    words = [x.strip() for x in li[i][wcol].split(' ')]
    nums = []
    for ac in addcols:
      if ac >= 0: 
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1] * 0.01)
        else: num = float(num)
      else: num = 1
      nums.append(num)
    for i in range(len(words)):
      if words[i] in result: result[words[i]] = [pair[0] + pair[1] for pair in zip(result[words[i]],nums)]
  array = []
  for word in result.keys():
    array.append([word] + result[word])
  return sorted(array,key=sortkey,reverse=True)
# Calculate a total sum for every column in addcols and for every word pair in wcol
# words do not need to be beside each other or in any particular order, so "buy a dog house", "good house for dog owners", "dog in my house" all go under "dog house"
def twograms(li,wcol,addcols,wlist=0,sortkey=lambda x:1,allindices=False):
  if wlist == 0: wlist = wordlist(li,wcol) 
  result = {}
  if allindices:
    for i in range(len(wlist)):
      for j in range(len(wlist)):
        if i != j: result[compose([wlist[i],wlist[j]])] = [0] * len(addcols)
  for i in range(len(li)):
    if i % int(len(li)/10) == (int(len(li)/10) - 1): print "Two grams: " + str(i) + " / " + str(len(li))
    words = [x.strip() for x in li[i][wcol].split(' ')]
    nums = []
    for ac in addcols:
      if ac >= 0: 
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1]) * 0.01
        else: num = float(num)
      else: num = 1
      nums.append(num)
    for i in range(len(words)):
      if words[i] in wlist:
        for j in range(i+1,len(words)):
          if words[j] in wlist:
            comb = compose([words[i],words[j]])
            if comb in result: result[comb] = [pair[0] + pair[1] for pair in zip(result[comb],nums)]
            elif allindices == False: result[comb] = nums
  array = []
  for words in result.keys():
    array.append([words] + result[words])
  return sorted(array,key=sortkey,reverse=True)
# Calculate a total sum for every column in addcols and for every word triplet in wcol (do not need to be beside each other or in any particular order)
# setting allindices to True slows down the calculation a lot but gives you a CSV with all possible combinations of words, making it convenient for
# working with the same word list on different data
def threegrams(li,wcol,addcols,wlist=0,sortkey=lambda x:1,allindices=False):
  if wlist == 0: wlist = wordlist(li,wcol) 
  result = {}
  if allindices:
    for i in range(len(wlist)):
        for j in range(len(wlist)):
          for k in range(len(wlist)):
              if i != j and i != k and j != k: result[compose([wlist[i],wlist[j],wlist[k]])] = [0] * len(addcols)
  for i in range(len(li)):
    if i % int(len(li)/10) == (int(len(li)/10) - 1): print "Three grams: " + str(i) + " / " + str(len(li))
    words = [x.strip() for x in li[i][wcol].split(' ')]
    nums = []
    for ac in addcols:
      if ac >= 0:
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1]) * 0.01
        else: num = float(num)
      else: num = 1
      nums.append(num)
    for i in range(len(words)):
      if words[i] in wlist:
        for j in range(i+1,len(words)):
          if words[j] in wlist:
            for k in range(j+1,len(words)):
              if words[k] in wlist:
                comb = compose([words[i],words[j],words[k]])
                if comb in result: 
                  result[comb] = [pair[0] + pair[1] for pair in zip(result[comb],nums)]
                elif allindices == False: result[comb] = nums
  array = []
  for words in result.keys():
    array.append([words] + result[words])
  return sorted(array,key=sortkey,reverse=True)
# Calculate a total sum for every column in addcols and for every word quadruplet in wcol
def fourgrams(li,wcol,addcols,wlist=0,sortkey=lambda x:1):
  if wlist == 0: wlist = wordlist(li,wcol) 
  result = {}
  for i in range(len(li)):
    if i % int(len(li)/10) == (int(len(li)/10) - 1): print "Four grams: " + str(i) + " / " + str(len(li))
    words = [x.strip() for x in li[i][wcol].split(' ')]
    nums = []
    for ac in addcols:
      if ac >= 0:
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1]) * 0.01
        else: num = float(num)
      else: num = 1
      nums.append(num)
    for i in range(len(words)):
      if words[i] in wlist:
        for j in range(i+1,len(words)):
          if words[j] in wlist:
            for k in range(j+1,len(words)):
              if words[j] in wlist:
                for l in range(k+1,len(words)):
                  if words[l] in wlist:
                    comb = compose([words[i],words[j],words[k],words[l]])
                    if comb in result:
                      result[comb] = [pair[0] + pair[1] for pair in zip(result[comb],nums)]
                    else: result[comb] = nums
                  
  array = []
  for words in result.keys():
    array.append([words] + result[words])
  return sorted(array,key=sortkey,reverse=True)
# Filters array, returning only the rows where column wcol of that row contains the query keywords (keywords can appear in any order)
# This and the other filters are useful for taking a list of entries and creating a list of only valid entries according to some validity characteristic
# eg:
# dog house, 15
# cat, 18
# dog, 33
# filter(li,0,'dog'):
# dog house, 15
# dog, 33
def filter(li,wcol,query):
  result = []
  for i in range(len(li)):
    words = [x.strip() for x in li[i][wcol].split(' ')]
    inlist = True
    queryarray = query.split(' ')
    if queryarray == ['']: queryarray = []
    for w in queryarray:
      if w not in words: inlist = False
    if queryarray == ['*']: inlist = len(li[i][wcol]) > 0
    if inlist: result.append(li[i])
  return result
# Filters array, requiring column wcol to exactly match query
def phrasefilter(li,wcol,query):
  result = []
  for i in range(len(li)):
    if li[i][wcol] == query: result.append(li[i])
  return result
# Filters array, requiring function func taken of the row to return True (or 1)
def funcfilter(li,func):
  result = []
  for i in range(len(li)):
    if func(li[i]): result.append(li[i])
  return result
# Adds up columns in addcols for a query matching keyfilter(li,wcol,query); can also be thought of as doing a single n-keyword match
# eg:
# dog, 25
# cat, 15
# dog, 75
# dog, 10
# horse, 55
# cat, 7
# search(li,0,[1],'dog') gives ['dog',110]
def search(li,wcol,addcols,query):
  result = [0] * len(addcols)
  for i in range(len(li)):
    words = [x.strip() for x in li[i][wcol].split(' ')]
    nums = []
    for ac in addcols:
      if ac >= 0:
        num = str(li[i][ac]).replace(',','').replace(' ','')
        if num == '': num = 0
        elif num[-1] == '%': num = float(num[:-1] * 0.01)
        else: num = float(num)
      else: num = 1
      nums.append(num)
    inlist = True
    queryarray = query.split(' ')
    if queryarray == ['']: queryarray = []
    for w in queryarray:
      if w not in words: inlist = False
    if queryarray == ['*']: inlist = len(li[i][wcol]) > 0
    if inlist:
      result = [pair[0] + pair[1] for pair in zip(result,nums)]
  return [query] + result
# Print a CSV from an array to stdout
def tochars(array,sp=','):
  string = ""
  for line in array: string += sp.join([str(x) for x in line]) + '\n'
  return string[:-1]
# Save an array to CSV
def save(f,array,sp=','):
  writeto = open(f,'w')
  writeto.write(tochars(array,sp))
  writeto.close()
# Compares keywords by two different parameters from two different lists. For example, li1 can be a list of how much money is spent (on addcol1) on a particular combination of keywords (on keycol1) and li2 can be a list of upgraded accounts with the search query they came from on keycol2, and addcol 2 can be left blank to default to -1 (each row is worth one point). Fourth column is statistical significance.
# Remember that you may have to filter the list yourself first
# Arguments:
# grams = 1 for single keywords, 2 for pairs, 3 for triplets and 4 for quadruplets
# li1, li2 = your two lists
# keycol1, keycol2 = where the keywords are located in those two lists
# addcol1, addcol2 = the columns of what you want to add up, eg. cost (set to -1 or leave blank to make it add 1 for each row)
# sortkey = function to sort results by (highest first)
# usestem = stem keywords
# sigtable = add ratio and significance to table
# invertratio = set ratio column to col1/col2 instead of col2/col1
# preformatted = li1 and li2 are already properly formatted
# justpreformat = convert li1 and li2 into twocolumns for comparison but don't go all the way
# wordlimit = limit search to some more common keywords for speedup purposes
# Example: list of customers, some upgraded, with originating keywords, and a list of how much you're paying for each search phrase
#
# customers.csv:
# Name, Keyword, Status
# Bob Jones, spreadsheet csv software, upgraded
# Matt Bones, csv python utils, free
# Army Drones, free spreadsheet, free
# Glenn Mitt, csv software, upgraded
# Pat Submitt, python utils software, upgraded
# Shawn Wit, python spreadsheet program, upgraded
#
# costs.csv:
# csv software, useless, and, irrelevant, data, 5.00, blah, blah
# python spreadsheet, useless, and, irrelevant, data, 2.50, blah, blah
# spreadsheet utils, useless, and, irrelevant, data, 10.00, blah, blah
# csv utils, useless, and, irrelevant, data, 1.50, blah, blah
# 
# Steps:
# 1. import spread (if not imported already)
# 2. upgrades = spread.filter(spread.load('customers.csv'),2,'upgraded')
# 3. costs = spread.load('costs.csv')
# 4. res = compare(1,costs,upgrades,0,1,5,invertratio=True)
# 5. spread.save('saved.csv',res)
#
# Res should look like:
#
# Keyword, Column 1, Column 2, Ratio, Significance
# spreadsheet, 12.50, 2, 6.25, -0.389
# utils, 11.50, 1, 11.50, -0.913
# csv, 7.50, 2, 3.75, 0.335
# python, 2.50, 2, 1.25, 2.031
#
# Or, if desired, you can:
# i1,i2 = compare(1,costs,upgrades,0,1,5,justpreformat=True)
# res1 = compare(1,i1,i2,0,1,5,invertratio=True,preformatted=True)
# res2 = compare(2,i1,i2,0,1,5,invertratio=True,preformatted=True)
# res3 = compare(3,i1,i2,0,1,5,invertratio=True,preformatted=True)
# res4 = compare(4,i1,i2,0,1,5,invertratio=True,preformatted=True)
#
# Note that significance is calculated based on col2/col1 regardless of invertratio, since getting 0 upgrades when you should have gotten 2 is not that unlikely, but calculating significance based on col1/col2 would give you infinity as infinity is infinitely far away from 0.5.
def compare(grams,li1,li2,keycol1,keycol2,addcol1=-1,addcol2=-1,sortkey=lambda x:x[1],usestem=True,sigtable=True,invertratio=False,preformatted=False,justpreformat=False,wordlimit=0):
  gramfuncs = [0,onegrams,twograms,threegrams,fourgrams]
  if preformatted == False:
    s1 = declutter(cols(li1,[keycol1,addcol1]),[1])
    print "Done decluttering/stemming: 1/4"
    s2 = declutter(cols(li2,[keycol2,addcol2]),[1])
    print "Done decluttering/stemming: 2/4"
    s1 = stem(s1,[0]) if usestem else declutter(s1,[0])
    print "Done decluttering/stemming: 3/4"
    s2 = stem(s2,[0]) if usestem else declutter(s2,[0])
    print "Done decluttering/stemming: 4/4"
  else: s1,s2 = li1,li2
  print "Printing sample of list 1"
  print s1[:10]
  print "Printing sample of list 2"
  print s2[:10]
  if justpreformat: return s1,s2
  while type(s1[0][1]) is str: s1.pop(0)
  while type(s2[0][1]) is str: s2.pop(0)
  print "Cleaned invalid rows"
  wl = sorted_wordlist(s1,0,[1])
  if wl.count('') > 0: blank = wl.pop(wl.index(''))
  print "Base wordlist length: " + str(len(wl)) + " ; Top ten: " + str(wl[:10])
  if wordlimit > 0 and wordlimit < len(wl):
    print "Shortening to " + str(wordlimit)
    wl = wl[:wordlimit]
  res1 = gramfuncs[grams](s1,0,[1],wl)
  print "Done search: 1/2"
  res2 = gramfuncs[grams](s2,0,[1],wl)
  print "Done search: 2/2"
  comb = sorted(splice(res1,res2,0,0),key=sortkey,reverse=True)
  if sigtable:
    tot1 = search(s1,0,[1],'')
    tot2 = search(s2,0,[1],'')
    ev = tot2[1]*1.0/tot1[1]
    print "Totals: " + str(tot1[1]) + ", " + str(tot2[1])
    for i in range(len(comb)): 
      comb[i].append(comb[i][2 - invertratio]*1.0/(comb[i][1 + invertratio] + 0.000001))
      comb[i].append((comb[i][2] - ev * comb[i][1])*1.0/(ev * comb[i][1] + 0.000001) ** 0.5)
    comb = [['Keyword','Column 1','Column 2','Ratio','Significance']] + comb
  else: comb = [['Keyword','Column 1','Column 2']] + comb
  print "Done"
  return comb
