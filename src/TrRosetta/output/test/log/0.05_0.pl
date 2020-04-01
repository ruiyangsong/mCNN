#!/usr/bin/perl -w
use strict;

print `hostname`;
print `date`;

my $bindir="/public/home/yangserver/trRosetta/bin";
my $datadir="/public/home/yangjy/for/sry/output/test";
my $pcut="0.05";
my $k="0";
my $tag="test_0.05_0";
my $user="yangjy";

if(!-s "$datadir/seq.npz" || !-s "$datadir/seq.rama.npz" || !-s "$datadir/seq.fasta")
{
    die "some input file missed\n";
}


my $workdir="/tmp/$user/$tag";

`mkdir -p $workdir`;
chdir "$workdir";
`rm -fr $workdir/*`;

if(!-s "$datadir/seq.npz" || !-s "$datadir/seq.rama.npz" || !-s "$datadir/seq.fasta")
{
    die "some input file missed\n";
}


`cp $datadir/seq.npz .`;
`cp $datadir/seq.fasta .`;
`cp $datadir/seq.rama.npz .`;


`/public/home/yangjy/for/sry/distogram2spline_rama.py  seq.npz seq.rama.npz seq.fasta 0` if(!-s "cst.txt");

 `/usr/bin/python /public/home/yangjy/for/sry/fold_from_tor_split.py $pcut $k`;

`cp $workdir/pose*.pdb $datadir/`;
`cp $workdir/*.sc $datadir/`;

`rm -fr $workdir`;
print "done\n";
print `date`;

