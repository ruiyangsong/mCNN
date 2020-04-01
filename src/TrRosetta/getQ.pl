#!/usr/bin/perl -w
use strict;

#my $user="yangji";    ##### Change this to your name 
my $user=`whoami`;    ##### Change this to your name 
$user =~ s/\n//;
my $MYMAX=50;        ###change this to the maximum number of jobs by you
my $TOTMAX=100;      ##change this to the maximum number of jobs by all users

while(1)
{
    my $count="";
    while($count eq "")
    {
	$count=`/usr/local/bin/qstat -u $user|wc -l`;
	$count =~ s/\s+//g;
    }
    
    my $q="default";   
    

    if($count<$MYMAX)
    {
	print "$q";
	exit(1);
    }
    else
    {
	$count="";
	while($count eq "")
	{
	    $count=`/usr/local/bin/qstat |wc -l`;
	    $count =~ s/\s+//g;
	}

	if($count<$TOTMAX)
	{
	    print "$q";
	    exit(1);	 
	}
	else
	{
	    sleep(30);
	}
    }   
}


