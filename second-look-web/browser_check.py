"""Verify minimal chat, real progress, benchmark values, and responsive layout."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit, urlunsplit
from playwright.sync_api import expect, sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--ui-only',action='store_true')
    args=parser.parse_args()
    parsed=urlsplit(args.url)
    args.url=urlunsplit((parsed.scheme,parsed.netloc,'','',''))
    root=Path(__file__).resolve().parent
    out=root/'docs/screenshots';out.mkdir(parents=True,exist_ok=True)
    errors=[];stages=set();results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda error:errors.append(str(error)))
        polls={};requests=[]
        if args.ui_only:
            def submit(route):
                values=route.request.post_data_json
                assert set(values)=={'question','review_mode'}
                assert values['review_mode'] in {'light','fusion'}
                requests.append(values);key=str(len(requests));polls[key]=0
                route.fulfill(status=202,json={'id':key,'status':'queued'})
            def result(route):
                key=route.request.url.rsplit('/',1)[-1];i=polls[key];polls[key]+=1
                if requests[int(key)-1]['question']=='Trigger interface error.':
                    route.fulfill(json={'status':'error','error':'Please try again.'})
                elif i<6:
                    route.fulfill(json={'status':'running','stage':['generating','diagnosing','paraphrasing'][i//2]})
                else:
                    route.fulfill(json={'status':'complete','result':{'final_answer':'**Thank you** for your help. <img src=x onerror="window.injected=true">','seconds':3,'truncated':False}})
            page.route('**/api/generate',submit)
            page.route('**/api/results/*',result)
        page.goto(args.url.rstrip('/')+'/human-scale-chatbot-education',wait_until='networkidle',timeout=60000)
        expect(page).to_have_title('Human-scale AI Chatbot')
        expect(page.locator('#connection')).to_have_text('Online',timeout=240000)
        expect(page.locator('#submit')).to_be_disabled()
        expect(page.locator('#review-mode')).to_have_value('light')
        assert page.locator('#review-mode option').all_inner_texts()==['Light','Fusion']
        page.screenshot(path=str(out/'human-scale-desktop.png'),full_page=True)
        page.get_by_role('button',name='About Human-scale AI',exact=True).click()
        expected=[['Configuration','Toxicity ↓','Jailbreak accuracy ↑'],['Light','0.0397 / 0.1750','70.71% / 87.62%'],['Fusion','0.1008 / 0.1981','95.95% / 87.38%']]
        expected_bbq=[
            ['Light, Correct','69.45','88.00','98.03'],['Light, Diagnose','67.45','72.67','80.92'],
            ['Fusion, Correct','56.91','79.33','81.58'],['Fusion, Diagnose','78.73','90.00','84.21']]
        measured_path=root/'docs/metrics-evaluation.json'
        if measured_path.exists():
            measured=json.loads(measured_path.read_text());assert measured['complete']
            scores={(r['setting'],r['mode'],r['benchmark']):r['score'] for r in measured['results']}
            if measured.get('evaluation_mode')=='off_the_shelf':
                expected=[expected[0]]+[[s.title(),f"{scores[s,'off_the_shelf','rtp']:.4f}",
                    f"{100*scores[s,'off_the_shelf','jailbreak']:.2f}%"] for s in ('light','fusion')]
                expected_bbq=[[s.title()]+[f"{100*scores[s,'off_the_shelf','bbq_'+c]:.2f}" for c in ('gender','nationality','disability')]
                    for s in ('light','fusion')]
                expect(page.locator('.metric-note')).to_contain_text('off-the-shelf correction')
            else:
                expected=[expected[0]]+[[s.title(),
                    ' / '.join(f"{scores[s,m,'rtp']:.4f}" for m in ('Correct','Diagnose')),
                    ' / '.join(f"{100*scores[s,m,'jailbreak']:.2f}%" for m in ('Correct','Diagnose'))]
                    for s in ('light','fusion')]
                expected_bbq=[[f'{s.title()}, {m}']+[f"{100*scores[s,m,'bbq_'+c]:.2f}" for c in ('gender','nationality','disability')]
                    for s in ('light','fusion') for m in ('Correct','Diagnose')]
            expect(page.locator('.metric-note')).to_contain_text('DeepSeek')
            expect(page.locator('#paper-metric-comparison')).to_be_visible()
        actual=[row.locator('th,td').all_inner_texts() for row in page.locator('#toxicity-metrics tr').all()]
        assert actual==expected,(actual,expected)
        assert [row.locator('th,td').all_inner_texts() for row in page.locator('#bbq-metrics tbody tr').all()]==expected_bbq
        assert page.locator('#about a').get_attribute('href')=='https://arxiv.org/html/2601.03079v7'
        page.screenshot(path=str(out/'human-scale-about.png'),full_page=True)
        page.keyboard.press('Escape')
        page.locator('#question').fill('Keyboard test');page.locator('#question').press('Shift+Enter')
        assert '\n' in page.locator('#question').input_value()
        page.locator('#question').fill('Write one short sentence thanking a colleague for their help.')
        started=time.monotonic();page.locator('#question').press('Enter')
        deadline=time.monotonic()+300
        progress_saved=False
        while time.monotonic()<deadline:
            turn=page.locator('.turn').last
            if turn.locator('.request-error').count():raise RuntimeError(turn.locator('.request-error').inner_text())
            if turn.locator('.steps').count():
                assert turn.locator('.steps li').all_inner_texts()==['1. Thinking','2. Aligning','3. Responding']
                active=turn.locator('.steps .active')
                if active.count():stages.add(active.inner_text())
                if not progress_saved:
                    page.screenshot(path=str(out/'human-scale-progress.png'),full_page=True);progress_saved=True
            if turn.locator('.answer').count():break
            page.wait_for_timeout(150)
        else:raise RuntimeError('Inference did not finish')
        elapsed=round(time.monotonic()-started,2)
        expect(page.locator('#new-chat')).to_be_enabled()
        assert turn.locator('.answer').inner_text().strip()
        assert turn.locator('.response-details,.review-text,.mode-badge,.download,.steps').count()==0
        assert turn.locator('img').count()==0
        assert page.evaluate('window.injected === undefined')
        results.append({'seconds_in_browser':elapsed,'answer':turn.locator('.answer').inner_text()})
        page.screenshot(path=str(out/'human-scale-response.png'),full_page=True)
        page.locator('#new-chat').click();expect(page.locator('#welcome')).to_be_visible()
        page.locator('.chat-link').first.click();assert page.locator('.answer').count()==1
        if args.ui_only:
            assert stages=={'1. Thinking','2. Aligning','3. Responding'}
            page.locator('#question').fill('Trigger interface error.');page.locator('#submit').click()
            expect(page.locator('.request-error')).to_be_visible();expect(page.locator('#new-chat')).to_be_enabled()
            page.locator('.edit-question').click();assert page.locator('#question').input_value()=='Trigger interface error.'
        for width in [390,320]:
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            bounds=page.locator('.composer').bounding_box();assert 0<=bounds['y'] and bounds['y']+bounds['height']<=844
            page.locator('#toggle-sidebar').click();expect(page.locator('#new-chat')).to_be_visible()
            page.locator('#new-chat').click();expect(page.locator('#welcome')).to_be_visible()
            page.get_by_role('button',name='About Human-scale AI',exact=True).click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(out/f'human-scale-about-{width}.png'),full_page=True)
            page.get_by_role('button',name='Close about',exact=True).click()
            page.screenshot(path=str(out/f'human-scale-mobile-{width}.png'),full_page=True)
            page.locator('#toggle-sidebar').click();page.locator('.chat-link').first.click()
        assert not errors,errors
        evidence={'url':args.url,'simulated_inference':args.ui_only,'browser_errors':errors,'stages_observed':sorted(stages),'metrics_verified':True,'private_details_visible':False,'mobile_widths':[390,320],'requests':results}
        (root/'docs'/('human-scale-ui-check.json' if args.ui_only else 'human-scale-live-check.json')).write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(evidence,indent=2),flush=True)
        browser.close()

if __name__=='__main__':main()
