from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=colors.HexColor('#142D4E'), spaceAfter=9))
styles.add(ParagraphStyle(name='SectionCustom', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#142D4E'), spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=9.3, leading=12.7, spaceAfter=6))
styles.add(ParagraphStyle(name='SmallCustom', fontName='Helvetica', fontSize=8, leading=10.5, textColor=colors.HexColor('#536477'), spaceAfter=5))
story=[]
def p(text, style='BodyCustom'):
    story.append(Paragraph(text, styles[style]))
def table(rows, widths):
    t=Table(rows, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#142D4E')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('LEADING',(0,0),(-1,-1),11),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('ALIGN',(1,0),(-1,-1),'RIGHT'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF3F8'),colors.white]),
        ('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#C7D2DF'))]))
    story.extend([t,Spacer(1,7)])
def plot(name):
    story.append(Image(str(ROOT/name),width=505,height=505*5/14))
    story.append(Spacer(1,3))
def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#C7D2DF'))
    canvas.line(45,38,A4[0]-45,38)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#536477'))
    canvas.drawString(45,25,'TT361 | Group 2 | Journal 7 | Student ID 36127')
    canvas.drawRightString(A4[0]-45,25,f'{doc.page}')
    canvas.restoreState()

p('Headline recommendation<br/>with multi-armed bandits','TitleCustom')
p('TT361 · Group 2 · Journal number 7 · Student ID: <b>36127</b>','SmallCustom')
p('Part B | Comparing four strategies','SectionCustom')
p('Each strategy serves 20,000 users in each of 20 runs. Agent seeds are 0-19 and environment seeds are 1000-1019, as in the starter. Tables and plots show averages across runs. Regret is the sum of true CTR gaps for the selected headlines: expected clicks lost, not observed missed clicks.','SmallCustom')
p('<b>True CTRs:</b> H0: 9.0%, H1: 2.7%, H2: 4.5%, H3: 5.7%, H4: 6.5%, H5: 5.4%. H0 is best; its gap over runner-up H4 is 2.5 percentage points.','SmallCustom')
table([['Strategy','Regret','Traffic on best'],['Uniform A/B test','673.3','16.7%'],['Epsilon-Greedy (epsilon = 0.1)','134.1','80.6%'],['UCB1 (c = 2)','459.0','38.0%'],['Thompson Sampling','70.1','89.9%']],[273,100,132])
plot('part_b.png')
p('1. Ranking and why the winner wins','SectionCustom')
p('From lowest to highest regret, the ranking is Thompson Sampling (70.1), epsilon-Greedy (134.1), UCB1 (459.0), then uniform A/B testing (673.3). Thompson Sampling draws one value from each Beta(alpha, beta) belief and selects the largest; clicks increase alpha and non-clicks increase beta. As evidence accumulates, inferior headlines become less likely to win these draws, so exploration naturally decreases and H0 receives 89.9% of traffic. Its regret is 47.7% below epsilon-Greedy and 89.6% below uniform allocation in these runs.')
p('2. Why fixed epsilon keeps accumulating regret','SectionCustom')
p('Even after H0 is reliably identified, epsilon-Greedy still explores on 10% of visits. Uniform exploration has mean CTR 0.05633, so its expected regret per additional visitor approaches 0.1 x (0.09 - 0.05633) = 0.003367. This contributes about 67.3 expected lost clicks per 20,000 users even with perfect exploitation, leaving a positive long-run slope. The measured regret of 134.1 is higher because the policy also loses clicks while learning which headline is best.')
p('3. Why UCB1 loses to epsilon-Greedy here','SectionCustom')
p('UCB1 adds sqrt(2 ln(t) / N_i) to each observed CTR, making its uncertainty bonus large relative to these small probabilities. At t = 20,000 and an illustrative equal allocation N_i = 20,000/6, the bonus is about 0.0771: larger than every true gap here (0.025-0.063). Equal bonuses would cancel in the ranking, but as H0 gets more traffic its bonus shrinks while less-used arms retain larger bonuses, repeatedly drawing traffic back to them. This conservative exploration helps explain UCB1\'s 38.0% best-headline share and 459.0 regret, compared with 80.6% and 134.1 for epsilon-Greedy.')

story.append(PageBreak())
p('Part C2 | The UCB1 constant','TitleCustom')
p('Same CTRs, horizon, run count and seeds as Part B; only c changes.','SmallCustom')
table([['Exploration constant','Regret','Traffic on best'],['c = 0.5','270.5','62.1%'],['c = 2 (standard)','459.0','38.0%'],['c = 5','544.4','28.5%']],[273,100,132])
plot('part_c2.png')
p('How the constant changes exploration','SectionCustom')
p('The selection score is the observed CTR plus sqrt(c ln(t) / N_i). Multiplying c by a factor multiplies the bonus by the square root of that factor. Relative to c = 2, c = 0.5 halves the bonus, whereas c = 5 makes it about 1.58 times as large. At the illustrative equal-allocation counts used in Part B, the bonuses are approximately 0.0385, 0.0771 and 0.1219 respectively. These are scale comparisons, not the actual bonus of every arm at the end of the simulation.')
p('The results follow this exploration pattern. With c = 0.5, UCB1 gives 62.1% of traffic to H0 and loses 270.5 expected clicks; with c = 5, H0 gets only 28.5% and regret reaches 544.4. Reducing c from 2 to 0.5 lowers regret by 188.5 clicks, or 41.1%, while increasing c from 2 to 5 raises regret by 85.4 clicks. The traffic plot shows the larger constants spreading more visits across inferior headlines, and the regret curves show the cost accumulating over the day.')
p('The risk of making c too small','SectionCustom')
p('A smaller constant works better in this experiment, but it reduces protection against misleading early observations. With rare clicks, the best headline can initially receive only non-clicks while an inferior headline gets a lucky click. If the bonus is too weak, the agent may spend much of a finite day exploiting the wrong headline before revisiting the best one. At c = 0, the policy becomes greedy after its initial round and can remain stuck; a small positive c still provides exploration, but recovery can be slow.')
p('Among the three tested constants, c = 0.5 is best, but its regret of 270.5 is still above both epsilon-Greedy (134.1) and Thompson Sampling (70.1). Thus this extension supports lowering UCB1\'s exploration constant for this scenario, not replacing Thompson Sampling with UCB1. The conclusion is specific to these CTRs and the 20,000-user horizon; other gaps, horizons or random outcomes could change the ranking. Twenty-run averages describe the observed experiment, and no confidence intervals or statistical-significance claim are provided.')

story.append(PageBreak())
p('Part D | Recommendation','TitleCustom')
p('To: Product manager, daily lead headline','SmallCustom')
p('Use Thompson Sampling','SectionCustom')
p('I recommend Thompson Sampling for selecting the daily lead headline. It initially tries different headlines, then directs more visitors to the ones supported by click data. It continues testing uncertain alternatives, but does not reserve a fixed fraction of visitors for random choices throughout the day.')
p('In the simulation, the best headline had a 9% click rate. Showing it to every visitor would produce 1,800 expected clicks over 20,000 visits, but a real system must first learn which headline is best. Thompson Sampling lost only 70.1 expected clicks during that learning process and sent 89.9% of visitors to the best headline. Its implied expected click total was approximately 1,729.9, compared with 1,665.9 for epsilon-Greedy, 1,341.0 for standard UCB1 and 1,126.7 for uniform allocation. These are averages based on expected regret, not recorded click totals or a guarantee of live performance.')
p('Why this choice fits the evidence','SectionCustom')
p('Thompson Sampling improved expected clicks by about 64 per 20,000 visits over epsilon-Greedy, the next-best tested strategy. Adjusting UCB1 helped, but even the best tested setting lost 270.5 expected clicks, substantially more than Thompson Sampling. For a daily headline slot with reasonably stable audience preferences, these results make Thompson Sampling the strongest starting choice.')
p('Where it could go wrong','SectionCustom')
p('The experiment assumed that click rates stayed constant. If breaking news or a change in the audience makes a different headline better during the day, ordinary Thompson Sampling can keep favoring the earlier winner because it retains all past observations. A newly popular headline may then receive too little traffic to reveal the change quickly. Before using the system in that setting, I would test a version that gradually forgets older data or resets its beliefs after a meaningful content or audience change; adaptation was not tested in the selected extension.')
p('Practical rollout','SectionCustom')
p('Start with a monitored rollout and track click rates by time of day alongside the traffic given to each headline. Initialize fresh beliefs for new headline candidates and investigate sudden changes in results. Keep editorial standards in the selection process: maximizing clicks alone does not ensure that a headline is accurate, useful or trustworthy.')
story.append(Spacer(1,12))
p('Implementation and reproducibility','SectionCustom')
p('The submitted Python file implements UCB1 with an initial visit to every arm and Beta-Bernoulli Thompson Sampling with Beta(1, 1) priors. Both return Python integers; Thompson Sampling updates the shared counts and running means before its belief parameters. All 12 supplied self-checks passed with STUDENT_ID = 36127. The supplied CTR generator, environments, simulation and plotting functions were preserved; Part C2 is enabled.','SmallCustom')
p('To reproduce: install numpy and matplotlib, then run python mab_assignment.py. The script prints the two result tables and saves part_b.png and part_c2.png in the current directory.','SmallCustom')

pdf=ROOT/'mab_report_36127.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=35,bottomMargin=50,title='Multi-Armed Bandits for Headline Recommendation - 36127',author='Student 36127').build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
import fitz
qa=ROOT/'tmp'/'pdfs'; qa.mkdir(parents=True,exist_ok=True)
doc=fitz.open(pdf)
print('Pages:',len(doc))
for i,page in enumerate(doc):
    page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(qa/f'page-{i+1}.png')
    print(i+1,len(page.get_text()),page.get_text()[:80].replace('\n',' / '))
