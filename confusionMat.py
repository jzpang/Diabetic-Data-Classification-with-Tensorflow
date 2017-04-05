import sys

def main():
	tn=0
	tp=0
	fp=0
	fn=0
	count=0
	file=sys.argv[1]
	f=open(file,"r")
	for line in f:
		print line
		count=count+1
		words=line.strip("\r\n").split(",")
		print words
		#"<30" pandas give 1, else 0

		if (words[-2]=="<30" and words[-1]=="1"):
			tp=tp+1
			
		if (words[-2]=="NO" and words[-1]=="1"):
			tn=tn+1

		if (words[-2]=="NO" and words[-1]=="0"):
			fn=fn+1

		if (words[-2]=="<30" and words[-1]=="0"):
			fp=fp+1

	print tp, tn, fp, fn
	print count
	r_tp=tp*1.0/(tp+fp)
	r_tn=tn*1.0/(tn+fn)
	r_fp=fp*1.0/(tp+fp)
	r_fn=fn*1.0/(tn+fn)

	print "tp"+": "+str(r_tp)
	print "tn"+": "+str(r_tn)
	print "fp"+": "+str(r_fp)
	print "fn"+": "+str(r_fn)


main()