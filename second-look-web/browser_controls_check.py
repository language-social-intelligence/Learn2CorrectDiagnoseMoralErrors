"""Check mode selection and New chat during an unfinished response."""
import argparse,json,sqlite3,time
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
from playwright.sync_api import sync_playwright,expect

p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8768');p.add_argument('--ui-only',action='store_true');args=p.parse_args()
u=urlsplit(args.url);origin=urlunsplit((u.scheme,u.netloc,'','',''));root=Path(__file__).resolve().parent
expected_model_id=json.loads((root/'assistant_config.json').read_text())['main_model']['model_id']
questions=['Write one short sentence thanking a colleague for their help.','Explain why ice floats on water in two sentences.']
submitted=[];jobs={};finished={};released=set();errors=[]
with sync_playwright() as playwright:
 browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000})
 page.on('pageerror',lambda e:errors.append(str(e)))
 def capture(response):
  if '/api/generate' in response.url and response.status==202:
   key=response.json()['id'];jobs[key]=response.request.post_data_json
  elif '/api/results/' in response.url and response.status==200:
   data=response.json()
   if data['status'] in ('complete','error'):finished[response.url.rsplit('/',1)[-1]]=data
 page.on('response',capture)
 def request_seen(request):
  if request.method=='POST' and request.url.endswith('/api/generate'):submitted.append(request.post_data_json)
 page.on('request',request_seen)
 if args.ui_only:
  def submit(route):
   key=str(len(submitted));route.fulfill(status=202,json={'id':key,'status':'queued'})
  def result(route):
   key=route.request.url.rsplit('/',1)[-1]
   if key in released:
    route.fulfill(json={'status':'complete','result':{'final_answer':f'Answer for {key}.','seconds':1,'truncated':False}})
   else:route.fulfill(json={'status':'running','stage':'diagnosing'})
  page.route('**/api/generate',submit);page.route('**/api/results/*',result)
 page.goto(origin+'/human-scale-chatbot-education',wait_until='networkidle')
 expect(page.locator('#connection')).to_have_text('Online',timeout=240000)
 mode=page.get_by_role('combobox',name='Alignment mode');expect(mode).to_have_value('light')
 page.locator('#question').fill(questions[0]);page.locator('#submit').click()
 expect(page.locator('.steps')).to_be_visible();expect(mode).to_be_disabled();expect(page.locator('#new-chat')).to_be_enabled()
 page.locator('#new-chat').click();expect(page.locator('#welcome')).to_be_visible();expect(page.locator('#question')).to_have_value('')
 mode.select_option('fusion');page.locator('#question').fill(questions[1]);page.locator('#submit').click()
 expect(page.locator('.steps')).to_be_visible();expect(mode).to_have_value('fusion')
 draft='Keep this draft while the other chat finishes.';page.locator('#question').fill(draft)
 def wait_finished(count):
  deadline=time.monotonic()+240
  while len(finished)<count and time.monotonic()<deadline:
   page.wait_for_timeout(100)
   for job in finished.values():
    if job['status']=='error':raise AssertionError(job)
  assert len(finished)>=count,finished
 released.add('1');wait_finished(1)
 expect(mode).to_have_value('fusion');expect(page.locator('#question')).to_have_value(draft)
 assert page.locator('.answer').count()==0,'Background answer leaked into the active chat.'
 page.get_by_role('button',name=questions[0],exact=True).click()
 expect(page.locator('.answer')).to_be_visible();expect(mode).to_have_value('light');expect(mode).to_be_enabled()
 page.get_by_role('button',name=questions[1],exact=True).click();expect(mode).to_have_value('fusion');expect(page.locator('#question')).to_have_value(draft)
 released.add('2');wait_finished(2);expect(page.locator('.answer')).to_be_visible();expect(mode).to_be_enabled()
 assert submitted==[{'question':questions[0],'review_mode':'light'},{'question':questions[1],'review_mode':'fusion'}],submitted
 evidence=[]
 for key,values in jobs.items():
  result=finished[key]['result'];assert set(result)=={'final_answer','seconds','truncated'}
  if not args.ui_only:
   with sqlite3.connect(root/'data/second-look.sqlite3') as db:
    saved=json.loads(db.execute('SELECT payload FROM jobs WHERE id=?',(key,)).fetchone()[0])
   assert saved['values']==values
   assert saved['result']['review_mode']==values['review_mode']
   assert saved['result']['main_model_id']==expected_model_id
  evidence.append({'mode':values['review_mode'],'result':result})
 page.locator('#new-chat').click();expect(page.locator('#welcome')).to_be_visible();expect(mode).to_have_value('fusion')
 out=root/'docs/screenshots';out.mkdir(parents=True,exist_ok=True)
 page.screenshot(path=str(out/'chat-controls-desktop.png'),full_page=True)
 for width in [390,320]:
  page.set_viewport_size({'width':width,'height':844})
  expect(mode).to_be_visible();mode.select_option('fusion')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  page.locator('#toggle-sidebar').click();page.locator('#new-chat').click()
  expect(page.locator('#welcome')).to_be_visible();expect(mode).to_have_value('fusion')
  expect(page.locator('#nav-backdrop')).to_be_hidden()
  page.screenshot(path=str(out/f'chat-controls-mobile-{width}.png'),full_page=True)
 assert not errors,errors
 report={'url':origin,'simulated_inference':args.ui_only,'new_chat_during_generation':True,'background_reply_isolated':True,'draft_preserved':True,'selection_preserved_per_chat':True,'both_modes_verified':True,'mobile_widths':[390,320],'browser_errors':errors,'requests':evidence}
 (root/'docs'/('chat-controls-ui-check.json' if args.ui_only else 'chat-controls-live-check.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True);browser.close()
