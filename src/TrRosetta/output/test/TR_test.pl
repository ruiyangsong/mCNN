#! /usr/bin/perl -w
use strict;


`hostname`=~/(\S+)/;
my $node=$1;
my $datetime=`date`;
my $path=`pwd`;

print "Hostname: $node\n";
print "Started : $datetime\n";
print "Path    : $path\n";




## Scores each residues as putative binding site residue based on JSD and 
## PsiBlast output
my $s = "test";
my $datadir = "/public/home/yangjy/for/sry/output/test";
my $user = "yangjy";

=pod
$s         ="TR000001"; #jobid
$datadir   ="/public/home/yangserver/trRosetta/output/$s";  #for seq.fasta
$user      ="yangserver";
=cut

######################################################
my $tag          ="TR_$s";
my $workdir      ="/tmp/$user/$tag";
my $qstat= "/usr/local/bin/qstat -f";
my $qsub= "/usr/local/bin/qsub";
if(!-s "$datadir/seq.fasta")
{
    die "seq.fasta missed\n";
}


`mkdir -p $workdir`;
chdir "$workdir";
`rm -fr $workdir/*`;

my $kmax=1;



my $bindir= "/public/home/yangserver/trRosetta/bin";
my $hhdir="/public//home/yangjy/software/hhsuite3/bin";
my $hhdb = "/library/uniclust30/uniclust30_2018_08"; #for hhblits

my $hmmdir= "/public/home/yangjy/software/hmmer/binaries";
my $uniref90="/library/database/uniref90/uniref90"; #for psiblast
my $uniref100="/library/database/uniref100/uniref100.fasta"; #for hmmsearch

my $deepdir= "/public/home/yangjy/software/ResNet/scripts";

my $diso_dir="/public/home/yangjy/software/DISOPRED";
my $itdir="/public/home/yangjy/software/I-TASSER4.4"; 

my $pythondir="/public/home/yangjy/anaconda/bin";

# Step 1. Run HHblits to generate four MSAs (only one for server)
# Step 2. Run hmmsearch against uniref to one MSA (skip for server)
# Step 3. Run Network to predict distance and orientation distributions
# Step 4. Run Rosetta to predict structure


my @probCuts=();

#push(@probCuts, 0.45);
#push(@probCuts, 0.35);
#push(@probCuts, 0.25);
#push(@probCuts, 0.15);
push(@probCuts, 0.05);





`cp $datadir/seq* .`;
my $seq="";
my @seqs=`cat seq.fasta`;
chomp(@seqs);
foreach my $line(@seqs)
{
    next if($line =~/^>/);
    $line =~ s/\s+//;
    $seq .= $line;
}
open(SEQ, ">seq.fasta");
print SEQ ">seq\n$seq\n";
close(SEQ);

my $len=&get_seq_len("seq.fasta");
my $ehour=&get_e_hours($len);



#print "run 1D...\n";
#&run_1D();

my @e=();
#push(@e, 1e-40);
#push(@e, 1e-10);
#push(@e, 1e-3);
#push(@e, 1);
my $evalue="0.001";

#foreach my $evalue(@e)
{
#    next if(-s "seq_$evalue.a3m");
if(!-s "seq.npz")
{
    print("running hhblits with $evalue\n");
    if(!-s "seq.a3m")
    {
        &run_hhblits("seq", $evalue);
        `cp seq.a3m $datadir/`;
    }
}


    print("running network\n");
   
    if(!-s "seq.cont")
    {
	&run_deep("seq.a3m", "seq");
	`cp seq.cont $datadir/`;
	`cp seq.npz $datadir/`;
	`cp seq.rama.npz $datadir/`;

#	last;
	
    }
}

my $prob=&top_prob("seq.cont", $len);

open(OUT, ">$datadir/top_prob.txt");
printf(OUT "%.3f\n", $prob);
close(OUT);


print "run rosetta\n";
&run_rosetta();

chdir $datadir;
&check_cen();

print "run refine\n";
&run_ref();
&check_ref();

&select_models();
&cal_cscore();




`rm -fr $workdir`;




sub cal_cscore
{

    my $pTM=`cat pTMscore.txt`; chomp($pTM);
    my $topP=`cat top_prob.txt`; chomp($topP);

    my $eTMscore=0.5096*$topP+0.5087*$pTM-0.096;

	$eTMscore=0.1 if($eTMscore<0.1);

    open(OUT, ">cscore.txt");
    printf(OUT "%.3f\n", $eTMscore);
    close(OUT);


}
sub check_ref
{
    my $tag="$s\_pose";

    while(1)
    {
	my $buff=`$qstat`;
	if($buff =~/$tag/)
	{
	    sleep(60);
	}
	else
	{
	    last;
	}
    }
}


sub get_seq_len
{
    my ($file)=@_;

    my @rst=`cat $file`;
    chomp(@rst);
    my $seq="";
    foreach my $r(@rst)
    {
        next if($r =~ /^>/);
        $r =~ s/\s+//mg;
        $seq .= $r;
    }

    my $len=length $seq;

    return $len;
}






sub top_prob
{
    my ($map, $len)=@_;

    die "$map missed\n" if(!-s $map);

    my %p=();
    my @pred=`cat $map`;chomp(@pred);
    my $pattern="";
    $pattern = '^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+(\S+)';

    foreach my $k(@pred)
    {
        if($k =~ /$pattern/)
        {
            my $r1=$1;my $r2=$2;my $s=$3;
            $s=0 if($s!~/\d+/);
            if($r1>$r2)
            {
                my $r=$r1;
                $r1=$r2; $r2=$r;
            }
            my $sep=$r2-$r1;
            if($sep>=12)
            {
                my $key=$r1 . "_" . $r2;
                $p{$key}=$s;
            }
        }
    }


    my @keys = sort{$p{$b} <=> $p{$a}} keys %p;

    my $cut=$len;
    my $count=0;
    my $TP=0;
    my $FP=0;
    my $sump=0;
    foreach my $k(@keys)
    {
        $count++;
        $sump +=$p{$k};
        last if($count>$cut);
    }


    return $sump/$cut;
}



sub run_rosetta
{
    chdir $datadir;
    #`$bindir/distogram2spline_rama.py  seq.npz seq.rama.npz seq.fasta` if(!-s "cst.txt");

    my $buff=`$qstat`;

    foreach my $pcut(@probCuts)
    {
	for(my $k=0; $k<$kmax; $k++)
	{
	    next if(-s "cen_$pcut\_$k.sc");
	    &submit_jobs($pcut, $k, $buff);
#exit;
	}
    }

}



sub submit_jobs
{
    my ($pcut, $k, $buff)=@_;
    my $scorefile="$datadir/cen_$pcut\_$k.sc";

    if(-s $scorefile)
    {
        my @rst=`cat $scorefile`;
        return if(@rst>=3);
    }

    my $tag="$s\_$pcut\_$k";
    return if($buff=~/$tag/);

    my $dir1="$datadir/log";
    `mkdir -p $dir1`;

    my $err="$dir1/err_$pcut\_$k";
    my $out="$dir1/out_$pcut\_$k";

    my $mod=`cat /public/home/yangjy/for/sry/scriptmod.pl`;
    $mod =~ s/\!DATADIR\!/$datadir/mg;
    $mod =~ s/\!PCUT\!/$pcut/mg;
    $mod =~ s/\!K\!/$k/mg;
    $mod =~ s/\!TAG\!/$tag/mg;
    $mod =~ s/\!USER\!/$user/mg;
    $mod =~ s/\!BINDIR\!/$bindir/mg;


    my $prog="$dir1/$pcut\_$k.pl";
    open(OUT, ">$prog");
    print OUT "$mod";
    close(OUT);
	`chmod 755 $prog`;

    my $walltime="walltime=48:00:00";

    #`$bindir/ros_q.pl $s`;
    #`$qsub -N $tag -o $out -e $err -l $walltime $prog`;
    `$prog`;

    print "$tag submitted\n";
}


sub check_cen
{
    my $done=0;

    my $buff="";
    #check if jobs are all done
    while(1)
    {
	$buff=`$qstat`;
	$done=1;
	foreach my $pcut(@probCuts)
	{
	    my $k=0;
	    for($k=0; $k<$kmax; $k++)
	    {
		my $tag="$s\_$pcut\_$k";
		if($buff=~/$tag/)
		{
		    $done=0;
		    last;
		}
		
	    }
	    if($done==0)
	    {
		last;
	    }
	}
	
	if($done==0)
	{
	    print "stage 1 is still running\n";
	    sleep(60);
	}
	else
	{
	    last;
	}
    }
    
    
    #check if models are generated sucesfully
    $done=0;
    foreach my $pcut(@probCuts)
    {
	my $k=0;
	for($k=0; $k<$kmax; $k++)
	{
	    my $flag=&check_jobs($pcut, $k);
	    last if($flag==0);
	}
	if($k==$kmax) {$done=1;}
    }
    
    if($done==0)
    {
	print "no stage1 jobs are running for $s, but some models are not generated sucesfully\n";
	return 0;
    }
    
}
   
sub run_ref
{
    my $buff=`$qstat`;
    chdir $datadir;
    
    #prepare for refining
    `cat cen*.sc >score.sc`;
    my $topN=3;
    
    my %all=&find_lowest_energy("score.sc", $topN);
    
    my @keys = keys %all;

    my $nn=@keys-1;
    for(my $i=0; $i<@keys; $i++)
    {
	my $k=$keys[$i];
	next if(!-s "$k.pdb");
	next if(-s "$k\_ref.pdb");
	
	my $tag="$s\_$k";
	next if($buff=~/$tag/);
	
	my $dir="$datadir/log";
	`mkdir -p $dir`;
	
	my $mod=`cat /public/home/yangjy/for/sry/refmod.pl`;
	$mod =~ s/\!DATADIR\!/$datadir/mg;
	$mod =~ s/\!MODEL\!/$k/mg;
	$mod =~ s/\!TAG\!/$tag/mg;
	$mod =~ s/\!USER\!/$user/mg;
	$mod =~ s/\!BINDIR\!/$bindir/mg;

	my $prog="$dir/$tag.pl";
	open(OUT, ">$prog");
	print OUT "$mod";
	close(OUT);


	my $err="$dir/err_$tag";
	my $out="$dir/out_$tag";

	`chmod 700 $prog`;
	my $walltime="walltime=48:00:00";

        #`$bindir/ros_q.pl $s`;
	if($i<$nn)
	{
	    `$qsub -N $tag -o $out -e $err -l $walltime $prog`;
	}
	else
	{
	    `$prog`;
	}
	print "$tag submitted\n";

    }

    return 1;
}



sub find_lowest_energy
{
    my ($scorefile, $top_N)=@_;

    return 0 if(!-s "$scorefile");

    my @rst=`cat $scorefile`; chomp(@rst);
    my %all=();

    foreach my $p(@probCuts)
    {
        my $pose="";
        my %batch=();
        foreach my $r(@rst)
        {
            if($r =~ /^(\S+).pdb\t+(\S+)/)
            {
                my $pose=$1;
                my $s=$2;
                my @a=split(/_/, $pose);
                next if($a[1]!=$p);

                $batch{$1}=$2;
            }
        }
        my @keys = sort {$batch{$a} <=> $batch{$b}} keys %batch;
        my $topN=$top_N; #the top number models from each prob cutoff

        $topN=@keys if($topN>@keys);
        for(my $i=0; $i<$topN; $i++)
        {
            my $k=$keys[$i];
            $all{$k}=$batch{$k};
        }
    }
    return %all;
}




sub select_models
{

    `cat ref*.sc >score_r.sc`;
    my $topN=3;
    
    my %all=&find_lowest_energy("score_r.sc", $topN);
    
    my @keys = sort {$all{$a} <=> $all{$b}} keys %all;
    if(@keys==0)
    {
	print "$s no model\n";
	next;
    }
    
    
    for(my $i=0; $i<$topN; $i++)
    {
	my $j=$i+1;
	`cp $keys[$i].pdb model$j.pdb`;
    }
    my $ave=&ave(\@keys);
    
    printf("pairwise TM-score %.3f\n", $ave);
    open(OUT, ">pTMscore.txt");
    printf OUT "%.3f\n", $ave;
    close(OUT);
}

sub ave
{
    my ($all_ref)=@_;
    my @all=@$all_ref;
    my $sum=0;
    my $count=0;
    my $N=10;

    if($N>@all)
    {
        $N=@all;
    }

    my @models=();
    for(my $i=0; $i<$N; $i++)
    {
        my $posei=$all[$i];
        $posei =~ s/ref/ref0/;
        push(@models, $posei);
    }

    for(my $i=0; $i<$N-1; $i++)
    {
        my $posei=$models[$i];
        for(my $j=$i+1; $j<$N; $j++)
        {
            my $posej=$models[$j];

            my $buff=`TMscore $posei.pdb $posej.pdb`;
            if($buff=~ /TM-score\s+=\s+(\S+)/)
            {
                my $tmscore=$1;
                $sum+=$tmscore;
                $count++;
            }
        }
    }


    if($count==0)
    {
        return 0;
    }
    else
    {
        return $sum/$count;
    }
}


sub check_jobs
{
    my ($pcut, $k)=@_;


    my $scorefile="cen\_$pcut\_$k.sc";

    if(-s $scorefile)
    {
        my @rst=`cat $scorefile`;
        return 1 if(@rst>=3);
    }
    else
    {
        return 0;
    }


}


sub run_1D
{
    if(!-s "seq.diso")
    {
	`$diso_dir/run_disopred.pl seq.fasta`;
	`cp seq.diso $datadir/`;
    }

    if(!-s "seq.dat")
    {
	`$itdir/PSSpred/PSSpred_yz.pl seq.fasta`;
	`cp seq.dat* $datadir/`;
    }
}


sub run_hhblits
{
    my ($jobid, $evalue)=@_;
    `$hhdir/hhblits -i $jobid.fasta -d $hhdb -oa3m $jobid.a3m -mact 0.35 -maxfilt 100000000 -neffmax 20 -cpu 4 -nodiff -realign_max 10000000 -maxmem 8 -n 4 -e $evalue`;
}


sub run_hmmsearch
{
    my ($jobid)=@_;

    `$hmmdir/hmmsearch --notextw --cpu 2 -o /dev/null -A $jobid.sto $jobid.hmm $uniref100`;
}


sub run_deep
{   
    my ($a3m, $id)=@_;
    
    
    #first try to predict with the original alignment
    my $count=`wc -l $a3m`;
    if($count =~ /^(\d+)/)
    {
        $count = $1;
    }
    my $n=$count;

    my $a3m1="tmp.a3m";

    if($n<=1)
    {   
	#`cat $a3m seq.fasta >> $a3m1`;
    }
    else
    {
	my $ncut0=50000;
	if($n>$ncut0)
	{ 
	    my $cn=0;
	    open(IN, "$a3m");
            open(OUT, ">tmp1.a3m");
            while(my $line=<IN>)
            {
                if($line=~/^>/)
                {
                    $cn++;
                    last if($cn>$ncut0);
                    print OUT "$line";
                    $line=<IN>;
                    print OUT "$line";
                }
            }
            close(OUT);
            close(IN);
            `mv tmp1.a3m $a3m1`;
	}
	else
	{
	    `cp $a3m $a3m1`;
	}
    } 

    while(1)
    {   
        print("predict with original msa\n");
        &pred($a3m1, $id);
        if(!-s "$id.cont")
        {                  
            my $size=10000;
            my @count=`cat $a3m1`;
            my $n=@count; $n=$n/2;
            
            if($n<=0)
            {   
                print "filtered alignment only contains single sequence, skip\n";
                return;
            }
            
            my $cn=0;
            if($n<=$size)
            {   
                $size=0.8*$n;
            }
            print "fail probably due to too many seqs ($n). Select the top $size, and run predict again\n";
            open(IN, "$a3m1");
            open(OUT, ">tmp1.a3m");
            while(my $line=<IN>)
            {   
                if($line=~/^>/)
                {   
                    $cn++;
                    last if($cn>$size);
                    print OUT "$line";
                    $line=<IN>;
                    print OUT "$line";
                }
            }
            close(OUT);
            close(IN);
            `mv tmp1.a3m $a3m1`;
        }
        else
        {   
            last;
        }
    }
    
    `rm -f $a3m1`;
}

sub pred
{
    my ($a3m, $id)=@_;

    #`pyhon $scdir/predict_rama.py $a3m $id.rama.npz`;

   `$pythondir/python $deepdir/predict_rama_v2.py $a3m $id.rama.npz`;
   `$pythondir/python $deepdir/predict_v2.py $a3m $id.npz $id.cont`;
}

sub get_e_hours
{
    my ($len)=@_;

    my $ehour=0.5;
    if($len<100)
    {
	$ehour=0.5;
    }
    elsif($len>=100 && $len<300)
    {
	$ehour=1;
    }
    elsif($len>=300 && $len<500)
    {
	$ehour=2;
    }
    elsif($len>=500 && $len<700)
    {
	$ehour=3;
    }
    elsif($len>=700)
    {
	$ehour=6;
    }
    elsif($len>=1000)
    {
        $ehour=24;
    }

    return $ehour;
}


sub print_status
{
    my ($st)=@_;

    my $html ='';
$html .= '
<head>
  <title>trRosetta modeling status</title>
  <link rel="icon" href="/images/nankai.jpg" type="images/jpg">
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <link rel="stylesheet" type = "text/css" href="/style/bootstrap-3.3.7/dist/css/bootstrap.css">
  <script src="/style/jquery-3.2.1.js"></script>
  <script type="text/javascript" src="/style/3Dmol.js"></script>
  <script src="/style/bootstrap-3.3.7/dist/js/bootstrap.js"></script>
</head>
<meta http-equiv="refresh" content="30">
<body>
  <nav class="navbar navbar-default" role="navigation">
    <div class="container-fluid">
      <div class = "navbar-inner">
        <ul class="nav navbar-nav">

          <li>
            <a href=http://yanglab.nankai.edu.cn/ target="_blank">
              <span class="glyphicon glyphicon-home"></span> Yang Lab
            </a>
          </li>

          <li>
            <a href=http://yanglab.nankai.edu.cn/trRosetta/ target="_blank">
              <span class="glyphicon glyphicon-inbox"></span> Server
            </a>
            </li>

            <li>
              <a href="http://yanglab.nankai.edu.cn/trRosetta/download/download.php" target="_blank">
                <span class="glyphicon glyphicon-download-alt"></span> Download
              </a>
            </li>


          <li>
            <a href="http://yanglab.nankai.edu.cn/trRosetta/example/" target="_blank">
              <span class="glyphicon glyphicon-file"></span> Example
            </a>
          </li>

          <li>
            <a href=http://yanglab.nankai.edu.cn/trRosetta/help/ target="_blank">
              <span class="glyphicon glyphicon-question-sign"></span> Help
            </a>
          </li>

          <li>
            <a href=http://yanglab.nankai.edu.cn/trRosetta/benchmark/ target="_blank">
              <span class="glyphicon glyphicon-book"></span> Benchmark
            </a>
          </li>

          <li>
            <a href=http://yanglab.nankai.edu.cn/trRosetta/news/ target="_blank">
              <span class="glyphicon glyphicon-comment"></span> News
            </a>
          </li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container">
';

$html .="

    <h3 align=\"center\"> trRosetta modeling status for job $s</h3>
This page is reloaded every minute and you will find the results automatically at this page once it is done.<br>
You can bookmark this page to check the results later.<br>
<a href=seq.fasta>Download the submitted sequence</a><br>
";

    my $h=$ehour;

    my $ehour0=$ehour - 0.2;
    if($st<=2)
    {
        $h=$ehour;
    }

    if($st==3)
    {
        $h=$ehour0;
    }

    if($st==4)
    {
	$h=$ehour0*0.5;	
    }

    if($st==5 || $st==6)
    {
        $h=0.1;
    }


    my %stat=(
        0=>" <b>Step 0</b>: Queue in cluster system <br>",
        1=>" <b>Step 1</b>: Generate multiple sequence alignment <br>",
        2=>" <b>Step 2</b>: Predict the distance and orientation distribution <br>",
        3=>" <b>Step 3</b>: Coarse-grained structure modeling by energy minimization <br>",
        4=>" <b>Step 4</b>: Full-atom structure refinement<br>",
        5=>" <b>Step 5</b>: Generate results page <br>",
        6=>" <b>Step 6</b>: Job completed and send notification email to user <br>",
        );

    $html .= "<br><br>Job status (estimated time needed to complete: <font color=\"red\"> $h hours</font>. The current stage is shown in red color):<br><br>\n";

    for(my $i=0; $i<=6; $i++)
    {
        my $line=$stat{$i};
        if($i==$st)
        {
            $html .= "<font color=\"red\"> $line </font>\n";
        }
        else
        {
            $html .= "$line\n";
        }
    }


    $html .= "
    </div>
  </body>
</html>
";


    open(ST, ">$datadir/index.html");
    print ST "$html";
    close(ST);

}
