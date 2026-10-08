#!/usr/bin/env python3
"""Fig. 1: the vehicle and the two-layer bounded-memory estimator. The figure is written as an SVG vector drawing
(1330-unit viewBox, Helvetica Neue) and rendered with cairosvg to PDF and PNG; all three files are written to figures/.
Layout: A and B side by side on top, C across the full width below, so that every label prints at 6.4 pt or more at the
7.16 in printed width (1 unit = 0.388 pt).
  A  the vehicle and its sensor suite (as in the earlier navigation-stack paper's Fig. 1B): event camera, 3-beam ToF rangefinder, IMU
  B  two-scale revisit course (loop A x2, then (B, A) x3): short-gap revisits fall inside the window, long-gap returns evict
  C  the two layers and their couplings: event camera (+ forward rangefinder beam) -> appearance key -> recognition (+ relative-displacement check) ->
     in-window (L1: window of W most-recently-used places, 0.25 m metric gate, relaxation) or evicted (sequence verifier
     -> fire the k cells); every closure injects the place's pattern into the grid attractor, which settles and is
     decoded; attitude (IMU + camera rotation rate) + modeled velocity + coarse visual translation drive path integration; the memory learns from the attractor and toward
     Layer 1's relaxed positions while the place is in the window, consolidates once at eviction, then is read-only.
Arrows are drawn as paths rather than typed as Unicode glyphs (Helvetica Neue lacks them under cairo). Mixed-style
(italic) runs inside anchored text are avoided because cairosvg places them incorrectly."""
import os, math
ROOT=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')); MAN=os.path.join(ROOT,'figures')
INK,GRY,MUT='#26313d','#8f99a2','#5c6b7a'; BLUE,ORA,GRN,VRM='#2a78d6','#E69F00','#009E73','#d4552b'; PUR='#5b4a86'
F="'Helvetica Neue',Arial,sans-serif"
out=[]
def esc(z):
    z=z.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    return z.replace('&lt;i&gt;','<tspan font-style="italic">').replace('&lt;/i&gt;','</tspan>')
def t(x,y,s,size=17,w=400,anchor='start',base='middle',lines=None,dy=20):
    """Text in ink (all figure text is black; colour is carried by boxes, arrows and marks)."""
    lines=[esc(l) for l in lines] if lines else None
    if lines:
        sp=''.join(f'<tspan x="{x}" dy="{0 if i==0 else dy}">{l}</tspan>' for i,l in enumerate(lines))
        out.append(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{w}" fill="{INK}" text-anchor="{anchor}" dominant-baseline="{base}">{sp}</text>')
    else: out.append(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{w}" fill="{INK}" text-anchor="{anchor}" dominant-baseline="{base}">{esc(s)}</text>')
def title(x,y,letter,s): out.append(f'<text x="{x}" y="{y}" font-family="{F}" font-size="24" fill="{INK}"><tspan font-weight="700">{letter}</tspan><tspan font-weight="500" dx="10">{esc(s)}</tspan></text>')
def box(x,y,w,h,fill='#eef1f4',stroke='#ccd4dc',rx=9,dash=None,sw=1.5):
    d=f' stroke-dasharray="{dash}"' if dash else ''; out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}></rect>')
def arrow(pts,color=INK,sw=1.8,dash=None,marker='ink'):
    d=f' stroke-dasharray="{dash}"' if dash else ''; p='M'+' L '.join(f'{x},{y}' for x,y in pts)
    out.append(f'<path d="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{d} marker-end="url(#arrow-{marker})"></path>')
def line(x1,y1,x2,y2,color=INK,sw=1.5,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''; out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}></line>')
def circ(cx,cy,r,fill,stroke=INK,sw=0.8): out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"></circle>')
def padlock(lx,ly): out.append(f'<rect x="{lx}" y="{ly+6}" width="14" height="10" rx="2" fill="{INK}"></rect><path d="M{lx+3},{ly+6} V{ly+3} a4,4 0 0 1 8,0 V{ly+6}" fill="none" stroke="{INK}" stroke-width="1.6"></path>')
W,H=1330,818
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{F}"><defs>')
for name,col in (('ink',INK),('gry',GRY),('vrm',VRM),('grn',GRN),('blu',BLUE),('ora',ORA),('pur',PUR)):
    out.append(f'<marker id="arrow-{name}" viewBox="0 0 10 10" markerWidth="7.2" markerHeight="7.2" refX="8" refY="5" orient="auto-start-reverse"><path d="M0.5,1 L9,5 L0.5,9 L3,5 Z" fill="{col}"></path></marker>')
out.append(f'<marker id="arrow-dim" viewBox="0 0 10 10" markerWidth="7" markerHeight="7" refX="8" refY="5" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="{GRY}" stroke-width="1.3"></path></marker>')
out.append('</defs>')
out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"></rect>')

# ============================ A: vehicle and sensors (top left) ============================
# Schematic insect-scale flyer and its sensor suite, drawn as in the earlier navigation-stack paper's Fig. 1B.
title(24,32,'A','Vehicle and sensors')
SIL=('<path d="M196,422 C170,398 138,382 128,392 C120,402 140,420 168,430 C182,434 192,430 196,422 Z" fill="#e2e8ee" fill-opacity="0.7" stroke="#a1acb6" stroke-width="1"></path>'
     '<path d="M200,416 C182,384 150,360 138,368 C128,376 146,404 176,420 C186,424 196,422 200,416 Z" fill="#e2e8ee" fill-opacity="0.85" stroke="#a1acb6" stroke-width="1.1"></path>'
     '<path d="M146,368 C162,388 182,408 198,417" fill="none" stroke="#a1acb6" stroke-width="0.8" opacity="0.7"></path>'
     '<path d="M84,438 C92,414 124,406 156,408 C194,410 208,422 208,438 C208,454 194,466 156,468 C124,470 92,462 84,438 Z" fill="#d3dbe2"></path>'
     '<path d="M85,434 L74,438 L85,442 Z" fill="#a1acb6"></path><ellipse cx="210" cy="436" rx="30" ry="28" fill="#d3dbe2"></ellipse><circle cx="247" cy="431" r="16.5" fill="#d3dbe2"></circle>'
     '<path d="M122,412 Q132,438 122,465" fill="none" stroke="#a1acb6" stroke-width="1" opacity="0.65"></path>'
     '<path d="M146,409 Q156,438 146,467" fill="none" stroke="#a1acb6" stroke-width="1" opacity="0.6"></path>'
     '<path d="M170,410 Q179,438 170,466" fill="none" stroke="#a1acb6" stroke-width="1" opacity="0.5"></path>'
     '<path d="M198,460 C192,476 184,486 176,498" fill="none" stroke="#a1acb6" stroke-width="1.5" stroke-linecap="round"></path>'
     '<path d="M214,464 C212,480 208,490 206,502" fill="none" stroke="#a1acb6" stroke-width="1.5" stroke-linecap="round"></path>'
     '<path d="M230,460 C236,476 240,486 244,498" fill="none" stroke="#a1acb6" stroke-width="1.5" stroke-linecap="round"></path>'
     '<path d="M258,420 C270,406 277,400 284,393" fill="none" stroke="#a1acb6" stroke-width="1.4" stroke-linecap="round"></path><circle cx="285" cy="392" r="2.2" fill="#a1acb6"></circle>'
     f'<line x1="248" y1="424" x2="248" y2="404" stroke="{BLUE}" stroke-width="1"></line><circle cx="248" cy="424" r="5.5" fill="{BLUE}" stroke="#fff" stroke-width="1.7"></circle>'
     f'<line x1="262" y1="438" x2="282" y2="438" stroke="{ORA}" stroke-width="1"></line><circle cx="262" cy="438" r="5.5" fill="{ORA}" stroke="#fff" stroke-width="1.7"></circle>'
     f'<circle cx="208" cy="436" r="5.5" fill="{GRN}" stroke="#fff" stroke-width="1.7"></circle>')
out.append(f'<g transform="translate(-30.8,-251) scale(0.82)">{SIL}</g>')          # silhouette at x 30-203, y 44-161
for yy,col,s in ((196,BLUE,'Event camera (DVS, 256 pixels, 90°)'),(220,ORA,'3-beam ToF rangefinder (0.1–2.83 m)'),(244,GRN,'IMU (accelerometer + pitch gyro)')):
    circ(38,yy,5.5,col,'#fff',1.1); t(52,yy,s,17)

# ============================ B: two-scale course (top middle) ============================
title(420,32,'B','Two-scale revisit course')
BX0,BY0,BS=430,50,190                          # arena box: left, top, side (2 m)
box(BX0,BY0,BS,BS,fill='#fbfcfd',stroke=INK,rx=0,sw=2.0)
S=BS/2.0; ax,ay=BX0+0.58*S,BY0+BS-0.58*S; rA=0.32*S; u=(math.cos(math.pi/4),math.sin(math.pi/4)); rB=0.50*S
bx,by=ax+(rA+rB)*u[0],ay-(rA+rB)*u[1]
out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{rA:.1f}" fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="6 4"></circle>')
out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{rB:.1f}" fill="none" stroke="{GRY}" stroke-width="1.6" stroke-dasharray="2 3"></circle>')
OBST=[]                                        # textured obstacles (x, y, w, h), mapped from the 2 m arena layout
for (lx,ly,lw,lh) in ((100,110,9,8),(250,100,7,10),(96,262,12,10),(258,270,9,12),(180,96,11,7),(262,190,7,9),(112,180,8,8),(230,284,10,8)):
    q=BS/210.0; ox,oy=BX0+(lx-70)*q,BY0+(ly-86)*q; OBST.append((ox,oy,lw*q,lh*q))
    out.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{lw*q:.1f}" height="{lh*q:.1f}" rx="1.5" fill="#c2cad1" stroke="{GRY}" stroke-width="1"></rect>')
tx,ty=ax+rA*u[0],ay-rA*u[1]                    # where the loops touch
circ(tx,ty,3.6,INK,stroke='#fff',sw=1.0)
th=math.radians(100); px,py=ax+rA*math.cos(th),ay-rA*math.sin(th); hd=th+math.pi/2   # the vehicle on loop A, facing along its path
p1=(px+9*math.cos(hd),py-9*math.sin(hd)); p2=(px+6*math.cos(hd+2.4),py-6*math.sin(hd+2.4)); p3=(px+6*math.cos(hd-2.4),py-6*math.sin(hd-2.4))
def _to_wall(x, y, ang):                       # distance from (x, y) along SVG direction ang to the first wall or obstacle
    dx, dy = math.cos(ang), -math.sin(ang); ts = []
    for wx in (BX0, BX0 + BS):
        if abs(dx) > 1e-9 and (wx - x) / dx > 0: ts.append((wx - x) / dx)
    for wy in (BY0, BY0 + BS):
        if abs(dy) > 1e-9 and (wy - y) / dy > 0: ts.append((wy - y) / dy)
    for (ox, oy, ow, oh) in OBST:               # slab test against each obstacle rectangle
        t0, t1 = 0.0, 1e9
        for o, d, lo, hi in ((x, dx, ox, ox + ow), (y, dy, oy, oy + oh)):
            if abs(d) < 1e-9:
                if not lo <= o <= hi: t0 = 1e9
                continue
            a_, b_ = sorted(((lo - o) / d, (hi - o) / d)); t0, t1 = max(t0, a_), min(t1, b_)
        if t0 <= t1 and t0 > 0: ts.append(t0)
    return min(ts)
FR = 30                                        # the camera's 90-degree field of view (wedge), drawn around the heading
fa, fb = hd - math.pi / 4, hd + math.pi / 4
out.append(f'<path d="M{px:.1f},{py:.1f} L{px+FR*math.cos(fa):.1f},{py-FR*math.sin(fa):.1f} A{FR},{FR} 0 0 0 {px+FR*math.cos(fb):.1f},{py-FR*math.sin(fb):.1f} Z" fill="{BLUE}" fill-opacity="0.16" stroke="{BLUE}" stroke-width="0.9" stroke-opacity="0.7"></path>')
for da in (-math.pi / 4, 0.0, math.pi / 4):    # the three rangefinder beams (0 and +-45 degrees), cast to a wall or obstacle
    ang = hd + da; L_ = _to_wall(px, py, ang); ex, ey = px + L_ * math.cos(ang), py - L_ * math.sin(ang)
    line(px, py, ex, ey, ORA, 1.3); circ(ex, ey, 2.6, ORA, '#fff', 0.8)
out.append(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} L{p3[0]:.1f},{p3[1]:.1f} Z" fill="{INK}" stroke="#fff" stroke-width="0.9"></path>')
line(BX0,BY0+BS+14,BX0+BS,BY0+BS+14,GRY,1.1); out[-1]=out[-1].replace('></line>',' marker-start="url(#arrow-dim)" marker-end="url(#arrow-dim)"></line>'); t(BX0+BS/2,BY0+BS+30,'2 m',17,anchor='middle')
t(ax+rA-4,ay+rA+2,'loop A',17,anchor='start'); t(bx,by,'loop B',17,anchor='middle')
NX=BX0+BS+22                                   # notes on the two loops, each with that loop's line style
line(NX,92,NX+26,92,INK,1.6,dash='6,4'); t(NX+34,92,'',17,lines=['loop A, flown repeatedly:','short-gap revisits,','served by Layer 1','(still in the window)'],dy=20)
line(NX,182,NX+26,182,GRY,1.6,dash='2,3'); t(NX+34,182,'',17,lines=['loop B, between laps of A:','long-gap returns to A,','served by Layer 2 (evicted)'],dy=20)

# ============================ key for C (top right) ============================
KX,KY=928,74
t(KX,KY-26,'Paths in C',17,w=700)
line(KX,KY,KX+34,KY,INK,1.8); t(KX+44,KY,'shared or symbolic path',17)
line(KX,KY+30,KX+34,KY+30,BLUE,2.4); t(KX+44,KY+30,'',17,lines=['learning while the place','is in the window'],dy=20)
line(KX,KY+80,KX+34,KY+80,VRM,2.4,dash='6,4'); t(KX+44,KY+80,'',17,lines=['evicted places: verification,','recall, consolidation'],dy=20)
padlock(KX+10,KY+118); t(KX+44,KY+128,'read-only after eviction',17)

# ============================ C: two-layer estimator (full width) ============================
out.append('<g transform="translate(0,28)">')    # C sits below the top row
title(24,292,'C','Two-layer estimator: bounded window and fixed-size synaptic memory')
R1,RH=310,58; RM=R1+RH/2                       # row 1: sensing, recognition, decision; attitude and velocity
box(30,R1,180,RH); circ(46,R1+15,5,BLUE,'#fff',1.1); t(120,R1+20,'',18,anchor='middle',lines=['Event camera','(DVS)'],dy=21)
arrow([(210,RM),(236,RM)])
box(238,R1,262,RH,fill='#eef5fb',stroke='#9cc3e6'); circ(254,R1+15,5,ORA,'#fff',1.1)   # the forward rangefinder beam also enters the key
t(369,R1+20,'',18,anchor='middle',lines=['Appearance key','(256-bit sparse binary)'],dy=21)
arrow([(500,RM),(526,RM)])
box(528,R1,238,RH,fill='#eef5fb',stroke='#9cc3e6'); t(647,R1+20,'',18,anchor='middle',lines=['Place recognition','+ displacement check'],dy=21)
arrow([(766,RM),(790,RM)])
DXc,DHW,DHH=872,82,31                          # decision diamond
out.append(f'<path d="M{DXc},{RM-DHH} L{DXc+DHW},{RM} L{DXc},{RM+DHH} L{DXc-DHW},{RM} Z" fill="#fbfcfd" stroke="{INK}" stroke-width="1.5"></path>'); t(DXc,RM,'in window?',17.5,anchor='middle')
box(1010,R1,156,RH); circ(1024,R1+15,5,GRN,'#fff',1.1); circ(1037,R1+15,5,BLUE,'#fff',1.1)   # IMU, plus the camera's rotation rate
t(1088,R1+20,'',18,anchor='middle',lines=['Attitude','(IMU, camera)'],dy=21)
arrow([(1166,RM),(1184,RM)])
box(1186,R1,132,RH,fill='#fdf1e0',stroke='#f0c07a'); circ(1198,R1+15,5,BLUE,'#fff',1.1); circ(1211,R1+15,5,ORA,'#fff',1.1)
t(1258,R1+20,'',18,anchor='middle',lines=['Velocity','(modeled)'],dy=21)
# yes -> Layer 1 (lands on the metric gate); no -> sequence verifier (evicted-place path, vermilion dashed)
GATEC=145
arrow([(DXc,RM+DHH),(DXc,388),(GATEC,388),(GATEC,500)],INK); t(DXc-8,374,'yes',17,anchor='end')   # head at the Layer 1 band
VFX,VFY,VFW,VFH=706,404,300,72
arrow([(DXc+DHW,RM),(986,RM),(986,VFY)],VRM,2.0,dash='6,4',marker='vrm'); t(994,388,'no (evicted)',17)
box(VFX,VFY,VFW,VFH,fill='#fdf1e0',stroke='#f0c07a'); t(VFX+VFW/2,VFY+19,'Sequence verifier',18,w=700,anchor='middle')
t(VFX+VFW/2,VFY+41,'',17,anchor='middle',lines=['route order + appearance margin','(no distance test)'],dy=19)
# --- Layer 1 band ---
L1X,L1Y,L1W,L1H=30,500,440,210
box(L1X,L1Y,L1W,L1H,fill='#f3effb',stroke='#cabce8',rx=11)
t(L1X+L1W/2,L1Y+21,'Layer 1: bounded symbolic window',18.5,w=700,anchor='middle')
box(50,540,190,52,fill='#ece5f7',stroke='#c7b6e6'); t(145,556,'',17,anchor='middle',lines=['metric gate','0.25 m'],dy=20)
arrow([(240,566),(262,566)])
box(264,540,190,52,fill='#ece5f7',stroke='#c7b6e6'); t(359,556,'',17,anchor='middle',lines=['relax','window'],dy=20)
y0=660; xs=[L1X+92+i*44 for i in range(7)]      # the window: chain of places, oldest left
for i in range(6): line(xs[i]+11,y0,xs[i+1]-11,y0,'#b9c2cb' if i<2 else INK,1.5)
for i,x in enumerate(xs):
    ev=i<2; circ(x,y0,11,'#f3effb' if ev else '#e9f1fb','#b9c2cb' if ev else INK,0.9 if ev else 1.2)
out.append(f'<path d="M{xs[6]},{y0-14} C {xs[6]-22},{y0-50} {xs[3]+22},{y0-50} {xs[3]},{y0-14}" fill="none" stroke="{PUR}" stroke-width="1.8" marker-end="url(#arrow-pur)"></path>')
t((xs[3]+xs[6])/2,y0-54,'loop edge',17,anchor='middle')
t((xs[0]+xs[1])/2,y0-26,'evicted',17,anchor='middle')
line(xs[2]-14,y0+22,xs[6]+14,y0+22,INK,1.1); line(xs[2]-14,y0+16,xs[2]-14,y0+22,INK,1.1); line(xs[6]+14,y0+16,xs[6]+14,y0+22,INK,1.1)
t((xs[2]+xs[6])/2,y0+38,'W most recently used places',17,anchor='middle')
# --- Layer 2 matrix ---
mx,my,mw,mh=760,500,190,210
box(mx,my,mw,mh,fill='#eef5fb',stroke='#9cc3e6',rx=9)
HI={1,4,7,9}                                    # a place's k random cells (k = 8 in the system; 4 drawn here for legibility)
for i in range(12):
    ry=my+14+i*15.8; hi=(i in HI)
    out.append(f'<rect x="{mx+14}" y="{ry}" width="{mw-28}" height="10" rx="2" fill="{BLUE if hi else "#dde3e9"}" stroke="#c8ced4" stroke-width="0.5"></rect>')
    if hi:   # sparse stored pattern: a few active grid cells per module inside the row
        for gx_ in (mx+26, mx+35, mx+84, mx+128, mx+137):
            out.append(f'<rect x="{gx_}" y="{ry+2.5}" width="5" height="5" rx="1" fill="#ffffff" opacity="0.85"></rect>')
t(mx+mw/2,my+mh+20,'Layer 2: fixed-size synaptic memory',18.5,w=700,anchor='middle')
t(mx+mw/2,my+mh+42,'each place: k = 8 random cells (4 drawn)',17,anchor='middle')
t(mx+mw/2,my+mh+63,'256 × 579 synapses (0.59 MB)',17,anchor='middle')
# verifier -> fire the k cells -> the place's rows
arrow([(856,VFY+VFH),(856,my)],VRM,2.0,dash='6,4',marker='vrm'); t(846,489,'fire the k cells',17,anchor='end')
# --- Layer 1 -> Layer 2 couplings ---
GXM=(L1X+L1W+mx)/2
arrow([(L1X+L1W,562),(mx,562)],BLUE,2.4,marker='blu'); t(GXM,524,'',17,anchor='middle',lines=['in window: learn toward','the relaxed position'],dy=20)
arrow([(L1X+L1W,646),(mx,646)],VRM,2.4,dash='6,4',marker='vrm'); t(GXM,668,'',17,anchor='middle',lines=['at eviction: consolidate,','then read-only'],dy=20)
padlock(GXM+72,680)
# --- grid attractor: three modules in one frame ---
GFX,GFY,GFW,GFH=1066,512,226,76
t(GFX,482,'Grid attractor',18.5,w=700); t(GFX+127,482,'(579 cells)',17)   # ends left of the path-integration arrow at x 1282
box(GFX,GFY,GFW,GFH,fill='#fbfcfd',stroke='#b9c2cb',rx=8,sw=1.2)
for k,n in enumerate((11,13,17)):
    x0=GFX+10+k*72
    out.append(f'<rect x="{x0}" y="{GFY+10}" width="56" height="56" rx="6" fill="#fff" stroke="{INK}" stroke-width="1.3"></rect>')
    for i in range(4):
        for j in range(4):
            hi=(i==2 and j==(1+k)%4)
            out.append(f'<rect x="{x0+5+i*12.5}" y="{GFY+15+j*12.5}" width="9.5" height="9.5" rx="2" fill="{BLUE if hi else "#dde3e9"}" stroke="#c8ced4" stroke-width="0.5"></rect>')
    t(x0+28,GFY+GFH+16,f'{n}×{n}',16.5,anchor='middle')
# velocity -> grid (path integration), into the frame
arrow([(1282,R1+RH),(1282,GFY)]); t(1272,430,'',17,anchor='end',lines=['path','integration'],dy=20)
# memory <-> grid: learning from the attractor (blue) and injection at every closure (ink)
arrow([(GFX,532),(mx+mw,532)],BLUE,2.4,marker='blu'); t((mx+mw+GFX)/2,520,'learn',17,anchor='middle')
arrow([(mx+mw,572),(GFX,572)],INK,2.2); t((mx+mw+GFX)/2,592,'',17,anchor='middle',lines=['inject','and settle'],dy=20)
# grid -> position estimate
PX,PY,PW,PH=1086,640,206,62
arrow([(GFX+GFW/2,GFY+GFH+26),(GFX+GFW/2,PY)]); box(PX,PY,PW,PH,fill='#e9f1fb',stroke='#a7c7e8')
t(PX+PW/2,PY+21,'Position estimate',18,w=700,anchor='middle'); t(PX+PW/2,PY+43,'(decoded from the grid)',17,anchor='middle')
out.append('</g>')
out.append('</svg>')
svg='\n'.join(out); os.makedirs(MAN,exist_ok=True); open(os.path.join(MAN,'fig1_architecture.svg'),'w').write(svg); print('svg written', len(svg),'bytes')
import cairosvg
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=os.path.join(MAN,'fig1_architecture.pdf'))
cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(MAN,'fig1_architecture.png'), output_width=2660)
from PIL import Image; _png = os.path.join(MAN, 'fig1_architecture.png'); Image.open(_png).save(_png, dpi=(300, 300))   # tag the resolution, as the other figures are
print('rendered ->', MAN)
