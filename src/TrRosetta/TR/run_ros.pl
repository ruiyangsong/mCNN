#! /usr/bin/perl -w
use strict;

## what you need to change
#my $user      = "yangjy";                 # your username
#my $datadir   = "/public/home/yangjy/for/sry/output";  # the datadir, where all targets are located
#my $list      = "list";         # the list of all proteins to predict,
#my $listdir   = "/public/home/yangjy/for/sry";     # where the list is located
my $user      = "sry";                 # your username
my $datadir   = "/public/home/sry/mCNN/dataset/TR/output";  # the datadir, where all targets are located
my $list      = "list";         # the list of all proteins to predict,
my $listdir   = "/public/home/sry/mCNN/dataset/TR";     # where the list is located
##end of change



#my $bindir    = "/public/home/yangjy/for/sry";     # where the COACH bin is located
my $bindir    = "/public/home/sry/mCNN/src/TrRosetta";     # where the COACH bin is located

open(LST,"<$listdir/$list");
while(my $line=<LST>)
{     
   # print "$line";
    chomp($line);
	
    my $name      = "";
    if($line =~/^(\S+)/)
    {
	$name=$1;
    }
    else	
    {
	exit;
    }
		

    my $rst=`/usr/local/bin/qstat -u $user |grep TR_$name`;

    next if(-s "$datadir/$name/model1.pdb");


    my $tag         = "TR_$name";	
    
    next if($rst =~ /$tag/); #to avoid resubmission

    my $outdir      = "$datadir/$name";
    `mkdir -p $outdir` if(!-d $outdir);    

    if(!-s "$outdir/seq.fasta")
    {
	die "seq.fasta missed\n";
    }	

    my $walltime    = "walltime=96:00:00,mem=4000m";
    my $prog_name   = "$tag.pl";	
    my $mod         = `cat $bindir/ROSmod.pl`;
    
    $mod =~ s/\!USER\!/$user/mg;
    $mod =~ s/\!DATADIR\!/$outdir/mg;
    $mod =~ s/\!TARGET\!/$name/mg;
    
    open(FH2,">$outdir/$prog_name");
    print FH2 "$mod";
    close(FH2);
    `chmod a+x $outdir/$prog_name`;	 
   `$bindir/getQ.pl`; 
    
    my $bsub=`/usr/local/bin/qsub -e $outdir/err_$tag -o $outdir/out_$tag -l $walltime -N $tag $outdir/$prog_name`;  	    
    chomp($bsub);
    if(length $bsub == 0)
    {
	last;
    }
    print "$tag was submitted!\n";	
}

close(LST);

