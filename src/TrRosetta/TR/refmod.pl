#!/usr/bin/perl -w
use strict;

print `hostname`;
print `date`;

my $bindir="!BINDIR!";
my $datadir="!DATADIR!";
my $tag="!TAG!";
my $user="!USER!";
my $model="!MODEL!";

if(!-s "$datadir/seq.npz" || !-s "$datadir/seq.rama.npz" || !-s "$datadir/seq.fasta")
{
    die "some input file missed\n";
}


if(!-s "$datadir/$model.pdb")
{
    die "$datadir/$model.pdb missed";
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
`cp $datadir/$model.pdb  .`;


`/public/home/yangjy/for/sry/distogram2spline_rama.py  seq.npz seq.rama.npz seq.fasta 1` if(!-s "cst.txt");


`/usr/bin/python /public/home/yangjy/for/sry/ref_split.py $model`;


`cp $workdir/*.pdb $datadir/`;
`cp $workdir/*.sc $datadir/`;

`rm -fr $workdir`;

print `date`;
print "done\n";
