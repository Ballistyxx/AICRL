** sch_path: /home/tim/gits/sky130_ef_ip__simple_por/xschem/sky130_ef_ip__simple_por.sch
.subckt sky130_ef_ip__simple_por vdd3v3 vdd1v8 porb_h porb_l por_l vss3v3 vss1v8
*.PININFO vdd3v3:B vss3v3:B porb_h:O porb_l:O por_l:O vdd1v8:B vss1v8:B
XC1 net9 vss3v3 sky130_fd_pr__cap_mim_m3_1 W=30 L=30 m=1
XC2 vss3v3 net9 sky130_fd_pr__cap_mim_m3_2 W=30 L=30 m=1
XM1 net3 net7 net5 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM2 net2 net3 vss3v3 vss3v3 sky130_fd_pr__nfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XR1 net4 vdd3v3 vss3v3 sky130_fd_pr__res_xhigh_po_0p69 L=500 mult=1 m=1
XM4 net5 net6 vdd3v3 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM5 net3 net3 vss3v3 vss3v3 sky130_fd_pr__nfet_g5v0d10v5 L=0.8 W=14 nf=7 m=1
XR2 vss3v3 net4 vss3v3 sky130_fd_pr__res_xhigh_po_0p69 L=150 mult=1 m=1
XM7 net2 net2 net1 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM8 net1 net1 vdd3v3 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=14 nf=7 m=1
XM10 net7 net4 vss3v3 vss3v3 sky130_fd_pr__nfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM9 net7 net7 net6 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM11 net6 net6 vdd3v3 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=16 nf=8 m=1
XM12 net8 net1 vdd3v3 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XM13 net9 net2 net8 vdd3v3 sky130_fd_pr__pfet_g5v0d10v5 L=0.8 W=2 nf=1 m=1
XR3 vss3v3 vss3v3 vss3v3 sky130_fd_pr__res_xhigh_po_0p69 L=25 mult=2 m=2
x2 net10 vss3v3 vss3v3 vdd3v3 vdd3v3 porb_h sky130_fd_sc_hvl__buf_8
x3 net10 vss1v8 vss1v8 vdd1v8 vdd1v8 porb_l sky130_fd_sc_hvl__buf_8
x4 net10 vss1v8 vss1v8 vdd1v8 vdd1v8 por_l sky130_fd_sc_hvl__inv_8
x5 net9 vss3v3 vss3v3 vdd3v3 vdd3v3 net10 sky130_fd_sc_hvl__schmittbuf_1
.ends
.end