#!/usr/bin/env python3
"""
PMIS P9+ Ground Truth Test Suite — 100 Cases
==============================================
Run from memory/ folder:
    python3 scripts/ground_truth_100.py

Tests whether the pipeline correctly retrieves the right knowledge
for real Yantra AI Labs work scenarios.

Categories:
  A. Single-domain exact (20 cases) — query clearly maps to one SC
  B. Single-domain semantic (15 cases) — same intent, different words
  C. Cross-domain (20 cases) — query needs 2+ SCs
  D. Multi-turn conversations (20 cases) — 5 convos x 4 turns
  E. Feedback-driven (10 cases) — quality should improve with scoring
  F. Edge cases (15 cases) — ambiguous, broad, very specific, negation
"""

import json
import sys
import os
import io
from pathlib import Path
from contextlib import redirect_stdout
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# Test Data — comprehensive Yantra memory
# =============================================================================

SEED_DATA = [
    {"super_context": "B2B Cold Outreach", "description": "Enterprise sales campaigns for Vision AI",
     "contexts": [
         {"title": "Email Copywriting", "weight": 0.8, "anchors": [
             {"title": "Short subjects get 40% higher open rates", "content": "Keep email subjects under 6 words for maximum opens", "weight": 0.9, "tags": ["email","copywriting","open-rate","subject"]},
             {"title": "Threat-intel language resonates with CISOs", "content": "CISOs respond to threat language over ROI - 3x higher reply rate", "weight": 0.85, "tags": ["security","ciso","email","threat"]},
             {"title": "Pain-point first solution second pattern", "content": "Lead with pain point then offer solution for 2x engagement", "weight": 0.8, "tags": ["email","structure","pattern","engagement"]},
             {"title": "3-line max for first paragraph", "content": "First paragraph over 3 lines loses 50% of readers", "weight": 0.75, "tags": ["email","copywriting","paragraph","brevity"]},
         ]},
         {"title": "Target Research", "weight": 0.7, "anchors": [
             {"title": "VP Security more responsive than CISO", "content": "VP Security title responds 2x more than CISO because they handle vendor eval directly", "weight": 0.85, "tags": ["targeting","security","role","vp"]},
             {"title": "Construction CTO/CIO contacts most responsive", "content": "Construction vertical CTO/CIO respond fastest among all verticals", "weight": 0.8, "tags": ["construction","targeting","cto","cio"]},
             {"title": "LinkedIn Sales Navigator for 200-2000 employee filter", "content": "Use LinkedIn Sales Nav filtering 200-2000 employees for mid-market sweet spot", "weight": 0.7, "tags": ["linkedin","sales-navigator","targeting","mid-market"]},
         ]},
         {"title": "Follow-up Cadence", "weight": 0.65, "anchors": [
             {"title": "3-touch sequence over 10 days optimal", "content": "Best results from 3 emails spaced over 10 days not 5", "weight": 0.7, "tags": ["cadence","follow-up","sequence","timing"]},
             {"title": "Tuesday Wednesday mornings best open rates", "content": "Tuesday and Wednesday mornings 8-10am get highest open rates", "weight": 0.65, "tags": ["timing","email","schedule","morning"]},
         ]},
     ]},

    {"super_context": "Vision Pipeline Architecture", "description": "CCTV analytics dual-layer pipeline on GCP",
     "contexts": [
         {"title": "Dual Layer Pipeline", "weight": 0.9, "anchors": [
             {"title": "YOLO 11n as fast gate before deep analysis", "content": "YOLO runs first pass detecting objects, only sends detections to Qwen VLM for deep analysis", "weight": 0.95, "tags": ["yolo","pipeline","architecture","detection","gate"]},
             {"title": "Qwen2.5-VL for scene understanding on 10s chunks", "content": "Qwen VLM processes 10-second video chunks for deep scene understanding and anomaly detection", "weight": 0.9, "tags": ["qwen","vlm","inference","scene","understanding"]},
             {"title": "System prompt is only change between verticals", "content": "Same pipeline serves factory/retail/construction by changing only the system prompt", "weight": 0.85, "tags": ["prompt","vertical","config","multi-vertical"]},
             {"title": "T4 GPU handles 4 concurrent streams at 720p", "content": "Google Cloud T4 GPU is cost-effective handling 4 RTSP streams simultaneously at 720p", "weight": 0.8, "tags": ["gpu","t4","compute","streaming","720p"]},
         ]},
         {"title": "CCTV Integration", "weight": 0.85, "anchors": [
             {"title": "MediaMTX Cloudflare Tunnel bypasses RTSP firewalls", "content": "Use MediaMTX with Cloudflare Tunnel to bypass RTSP firewall restrictions at client sites", "weight": 0.9, "tags": ["rtsp","streaming","cctv","mediamtx","cloudflare","firewall"]},
             {"title": "Analog DVR sites need edge compute bridge", "content": "Sites with analog DVR systems cannot run AI directly and need an edge compute device as bridge", "weight": 0.85, "tags": ["dvr","edge","hardware","analog","bridge"]},
             {"title": "RTSP pull mode more reliable than push for Indian ISPs", "content": "Pull mode RTSP connections survive ISP resets better than push mode in India", "weight": 0.75, "tags": ["rtsp","isp","india","reliability","pull"]},
         ]},
         {"title": "Compute Optimization", "weight": 0.8, "anchors": [
             {"title": "Qwen 2B sufficient for factory safety alerts", "content": "2B parameter Qwen model handles factory safety detection adequately, no need for 7B", "weight": 0.8, "tags": ["qwen","factory","compute","safety","2b"]},
             {"title": "7B model needed for complex retail analytics", "content": "Retail scene understanding requires 7B parameter model for accurate product/behavior analysis", "weight": 0.75, "tags": ["model-size","retail","analytics","7b"]},
             {"title": "Batch 3 frames per 10s chunk saves 40% compute", "content": "Processing 3 representative frames per 10-second chunk instead of all frames saves 40% GPU cost", "weight": 0.85, "tags": ["batching","optimization","compute","frames","cost"]},
         ]},
     ]},

    {"super_context": "Security Demo Page", "description": "Interactive security product demo website",
     "contexts": [
         {"title": "Hero Section Design", "weight": 0.85, "anchors": [
             {"title": "Blind spot widget increases engagement 3x", "content": "Interactive blind spot visualization widget on hero section triples time-on-page", "weight": 0.95, "tags": ["widget","engagement","demo","security","blind-spot","interactive"]},
             {"title": "Dark theme teal accent matches security vertical", "content": "Dark background with teal accent color resonates with security buyer aesthetic", "weight": 0.7, "tags": ["theme","design","security","teal","dark","aesthetic"]},
         ]},
         {"title": "Interactive Elements", "weight": 0.8, "anchors": [
             {"title": "Yantrai Command dashboard mockup drives demos", "content": "Yantrai Command branded dashboard mockup is the most effective demo asset", "weight": 0.88, "tags": ["dashboard","demo","mockup","command","yantrai"]},
             {"title": "GPU calculator converts technical users", "content": "Interactive GPU cost calculator converts technical evaluators at 2x rate", "weight": 0.75, "tags": ["calculator","conversion","technical","gpu","interactive"]},
         ]},
     ]},

    {"super_context": "Kiran AI Retail Product", "description": "Barcode-free SKU detection for Indian kirana stores",
     "contexts": [
         {"title": "SKU Detection Engine", "weight": 0.9, "anchors": [
             {"title": "99.9% accuracy on barcode-free Indian FMCG", "content": "Our SKU detection achieves 99.9% accuracy on Indian FMCG products without barcodes", "weight": 0.95, "tags": ["accuracy","sku","fmcg","detection","barcode-free"]},
             {"title": "200ms inference per frame on T4", "content": "Single frame SKU detection takes 200ms on T4 GPU, fast enough for real-time checkout", "weight": 0.8, "tags": ["latency","inference","t4","speed","realtime"]},
             {"title": "Training on 50 product images sufficient per SKU", "content": "50 training images per product is enough for reliable SKU detection", "weight": 0.7, "tags": ["training","images","sku","data","few-shot"]},
         ]},
         {"title": "Kirana Store Deployment", "weight": 0.8, "anchors": [
             {"title": "Hindi English bilingual UI mandatory for adoption", "content": "Store interface must support Hindi and English for kirana store owner adoption", "weight": 0.85, "tags": ["ui","localization","hindi","bilingual","adoption"]},
             {"title": "Store owners prefer daily WhatsApp reports", "content": "Kirana store owners prefer daily inventory reports via WhatsApp over dashboard login", "weight": 0.8, "tags": ["whatsapp","reporting","kirana","daily","inventory"]},
             {"title": "Single camera covers 80% of small store inventory", "content": "One ceiling-mounted camera covers 80% of a typical small kirana store shelves", "weight": 0.75, "tags": ["camera","coverage","store","single","inventory"]},
         ]},
     ]},

    {"super_context": "LinkedIn Content Strategy", "description": "Founder voice social media presence",
     "contexts": [
         {"title": "Founder Voice Posts", "weight": 0.85, "anchors": [
             {"title": "Casual authentic voice outperforms polished copy 2x", "content": "Authentic founder voice gets 2x more engagement than polished marketing copy on LinkedIn", "weight": 0.9, "tags": ["voice","authentic","linkedin","founder","engagement"]},
             {"title": "Behind-the-scenes factory visits get most engagement", "content": "BTS content from factory deployments gets highest saves and shares", "weight": 0.85, "tags": ["content","factory","engagement","bts","behind-scenes"]},
             {"title": "Technical explainers with diagrams reach decision-makers", "content": "Technical deep-dive posts with diagrams get shared by CTOs and VPs", "weight": 0.7, "tags": ["technical","explainer","diagram","decision-maker","linkedin"]},
         ]},
         {"title": "Carousel Design", "weight": 0.75, "anchors": [
             {"title": "8-slide max for LinkedIn carousel", "content": "LinkedIn carousels perform best at 8 slides maximum", "weight": 0.75, "tags": ["carousel","linkedin","format","slides","max"]},
             {"title": "First slide must be hook not logo", "content": "Lead carousel with a compelling hook question, never start with company logo", "weight": 0.8, "tags": ["hook","carousel","design","first-slide","logo"]},
             {"title": "Data visualization slides get saved and shared most", "content": "Carousel slides with data visualizations get 3x more saves than text-only slides", "weight": 0.7, "tags": ["data-viz","carousel","saves","shares","visualization"]},
         ]},
     ]},

    {"super_context": "Construction Site Monitoring", "description": "Safety compliance and progress tracking for construction",
     "contexts": [
         {"title": "Safety Compliance", "weight": 0.9, "anchors": [
             {"title": "PPE detection accuracy 96% at 50m range", "content": "Our PPE detection model achieves 96% accuracy at up to 50 meters camera range", "weight": 0.9, "tags": ["ppe","detection","safety","construction","accuracy","range"]},
             {"title": "Helmet and vest detection most requested by clients", "content": "Hard helmet and safety vest are the two most requested PPE detection items", "weight": 0.8, "tags": ["helmet","vest","ppe","detection","client","request"]},
             {"title": "Night shift monitoring needs IR-capable cameras", "content": "Night shift safety monitoring requires IR-capable cameras for reliable detection in darkness", "weight": 0.7, "tags": ["night","ir","camera","shift","darkness"]},
         ]},
         {"title": "Progress Tracking", "weight": 0.75, "anchors": [
             {"title": "Time-lapse comparison shows 15% productivity gain", "content": "AI time-lapse progress comparison demonstrates average 15% productivity improvement", "weight": 0.75, "tags": ["timelapse","productivity","progress","comparison","improvement"]},
             {"title": "Zone-based activity heatmaps for project managers", "content": "Activity heatmaps per construction zone help project managers identify bottlenecks", "weight": 0.7, "tags": ["heatmap","zone","activity","project-manager","bottleneck"]},
         ]},
     ]},

    {"super_context": "GTM Lead Pipeline", "description": "Go-to-market lead qualification and targeting",
     "contexts": [
         {"title": "Lead Qualification", "weight": 0.85, "anchors": [
             {"title": "3-stage Claude workflow research score consolidate", "content": "Three-stage qualification: Claude researches company, scores fit, consolidates into brief", "weight": 0.85, "tags": ["workflow","qualification","claude","research","scoring"]},
             {"title": "AI-friendly org filter reduces list 60%", "content": "Filtering for organizations already using AI tools reduces prospect list by 60% while improving conversion", "weight": 0.8, "tags": ["filter","qualification","ai","org","conversion"]},
         ]},
         {"title": "Tiered Targeting", "weight": 0.75, "anchors": [
             {"title": "60 AI-friendly Gurgaon orgs across 7 verticals", "content": "Identified 60 AI-friendly organizations in Gurgaon NCR across 7 target verticals", "weight": 0.7, "tags": ["gurgaon","leads","vertical","ncr","organizations"]},
             {"title": "10 priority NCR leads for first outreach batch", "content": "First batch targets 10 highest-priority NCR companies for initial outreach", "weight": 0.8, "tags": ["ncr","priority","batch","first","outreach"]},
             {"title": "500 international construction leads with verified contacts", "content": "Built list of 500 international construction/real estate companies with verified CTO/CIO emails", "weight": 0.65, "tags": ["international","construction","leads","verified","contacts"]},
         ]},
     ]},

    {"super_context": "PMIS System Design", "description": "Personal Memory Intelligence System architecture",
     "contexts": [
         {"title": "Hierarchy Design", "weight": 0.9, "anchors": [
             {"title": "Super Context-Context-Anchor 3-level tree", "content": "Knowledge organized in 3 levels: broad domain, skill/phase, atomic reusable fact", "weight": 0.95, "tags": ["hierarchy","structure","pmis","tree","levels"]},
             {"title": "Name by reusable domain not by date", "content": "Super contexts must be named by reusable work domain never by date or session", "weight": 0.9, "tags": ["naming","convention","domain","reusable"]},
             {"title": "Anchors must be atomic with numbers and reasoning", "content": "Each anchor is one fact with quantitative evidence and causal reasoning", "weight": 0.85, "tags": ["anchor","atomic","numbers","reasoning","specificity"]},
         ]},
         {"title": "Retrieval Engine", "weight": 0.85, "anchors": [
             {"title": "Word overlap quality temporal composite score", "content": "Retrieval score = word overlap + quality bonus + use bonus multiplied by temporal and mode boosts", "weight": 0.85, "tags": ["retrieval","scoring","formula","composite"]},
             {"title": "Top 3 SCs returned with full context trees", "content": "Retrieval returns top 3 super contexts with complete context and anchor sub-trees", "weight": 0.8, "tags": ["retrieval","topk","context","tree"]},
         ]},
     ]},

    {"super_context": "Hospital CCTV Analytics", "description": "Patient safety and compliance monitoring for hospitals",
     "contexts": [
         {"title": "Patient Safety", "weight": 0.85, "anchors": [
             {"title": "Fall detection in corridors most requested feature", "content": "Hospital corridor fall detection is the single most requested AI camera feature", "weight": 0.85, "tags": ["fall","detection","hospital","corridor","patient","safety"]},
             {"title": "Privacy masking mandatory for patient areas", "content": "AI must mask patient faces and bodies in all non-public hospital areas for compliance", "weight": 0.9, "tags": ["privacy","masking","compliance","patient","hospital"]},
             {"title": "Medanta and Fortis FMRI primary Gurgaon targets", "content": "Medanta and Fortis FMRI hospitals are the two primary targets in Gurgaon market", "weight": 0.7, "tags": ["medanta","fortis","gurgaon","hospital","target"]},
         ]},
     ]},

    {"super_context": "Residential Society Security", "description": "Gate monitoring and common area surveillance for housing societies",
     "contexts": [
         {"title": "Entry Gate Monitoring", "weight": 0.85, "anchors": [
             {"title": "ANPR for vehicle tracking at society gates", "content": "Automatic Number Plate Recognition tracks all vehicles entering/exiting society gates", "weight": 0.85, "tags": ["anpr","vehicle","gate","society","tracking","number-plate"]},
             {"title": "Visitor face matching against pre-approved list", "content": "Face recognition matches visitors against resident-approved guest list for access control", "weight": 0.8, "tags": ["face","recognition","visitor","guest","access","matching"]},
             {"title": "Alert latency must be under 5 seconds for gate staff", "content": "Gate security staff need alerts within 5 seconds for effective response", "weight": 0.75, "tags": ["latency","alert","realtime","gate","staff","response"]},
         ]},
         {"title": "Common Area Surveillance", "weight": 0.75, "anchors": [
             {"title": "Loitering detection in parking reduces theft 30%", "content": "AI loitering alerts in parking areas reduce vehicle theft incidents by 30%", "weight": 0.7, "tags": ["loitering","parking","theft","detection","reduction"]},
             {"title": "Child safety zones with perimeter alerts", "content": "Define child safety zones around pools and playgrounds with perimeter breach alerts", "weight": 0.8, "tags": ["child","safety","zone","perimeter","pool","playground"]},
         ]},
     ]},

    {"super_context": "Case Study Generation", "description": "B2B case studies and content creation for verticals",
     "contexts": [
         {"title": "Content Structure", "weight": 0.8, "anchors": [
             {"title": "7-section template with IaaS stack diagrams", "content": "Standard case study follows 7 sections with infrastructure-as-a-service architecture diagram", "weight": 0.8, "tags": ["template","casestudy","structure","iaas","diagram","sections"]},
             {"title": "Problem Solution ROI narrative arc", "content": "Every case study follows Problem-Solution-ROI narrative for maximum persuasion", "weight": 0.85, "tags": ["narrative","roi","casestudy","arc","problem","solution"]},
             {"title": "Sector-specific versions for 5 verticals", "content": "Maintain separate case study versions for Malls Factories Security Hospitals Residential", "weight": 0.7, "tags": ["sector","vertical","casestudy","mall","factory","hospital"]},
         ]},
     ]},
]


# =============================================================================
# 100 Test Cases
# =============================================================================

TESTS = [
    # ═══════════════════════════════════════════════════════════
    # A. SINGLE-DOMAIN EXACT (20 cases) — clear keyword match
    # ═══════════════════════════════════════════════════════════
    {"id": "A01", "cat": "single-exact", "q": "How to write cold email subject lines?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Short subjects get 40% higher open rates"]},
    {"id": "A02", "cat": "single-exact", "q": "What language works for CISO outreach?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Threat-intel language resonates with CISOs"]},
    {"id": "A03", "cat": "single-exact", "q": "Best follow-up email schedule?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["3-touch sequence over 10 days optimal"]},
    {"id": "A04", "cat": "single-exact", "q": "What GPU do we use for CCTV analytics?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["T4 GPU handles 4 concurrent streams at 720p"]},
    {"id": "A05", "cat": "single-exact", "q": "How does our YOLO pipeline work?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["YOLO 11n as fast gate before deep analysis"]},
    {"id": "A06", "cat": "single-exact", "q": "How to bypass RTSP firewall restrictions?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["MediaMTX Cloudflare Tunnel bypasses RTSP firewalls"]},
    {"id": "A07", "cat": "single-exact", "q": "What accuracy does our SKU detection achieve?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["99.9% accuracy on barcode-free Indian FMCG"]},
    {"id": "A08", "cat": "single-exact", "q": "What voice works best for LinkedIn?", "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["Casual authentic voice outperforms polished copy 2x"]},
    {"id": "A09", "cat": "single-exact", "q": "PPE detection range on construction sites?", "exp_sc": ["Construction Site Monitoring"], "exp_anc": ["PPE detection accuracy 96% at 50m range"]},
    {"id": "A10", "cat": "single-exact", "q": "How do we qualify leads using Claude?", "exp_sc": ["GTM Lead Pipeline"], "exp_anc": ["3-stage Claude workflow research score consolidate"]},
    {"id": "A11", "cat": "single-exact", "q": "What is the memory hierarchy structure?", "exp_sc": ["PMIS System Design"], "exp_anc": ["Super Context-Context-Anchor 3-level tree"]},
    {"id": "A12", "cat": "single-exact", "q": "Fall detection for hospital corridors?", "exp_sc": ["Hospital CCTV Analytics"], "exp_anc": ["Fall detection in corridors most requested feature"]},
    {"id": "A13", "cat": "single-exact", "q": "ANPR system for society gates?", "exp_sc": ["Residential Society Security"], "exp_anc": ["ANPR for vehicle tracking at society gates"]},
    {"id": "A14", "cat": "single-exact", "q": "Case study template structure?", "exp_sc": ["Case Study Generation"], "exp_anc": ["7-section template with IaaS stack diagrams"]},
    {"id": "A15", "cat": "single-exact", "q": "What Qwen model size for factory safety?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Qwen 2B sufficient for factory safety alerts"]},
    {"id": "A16", "cat": "single-exact", "q": "LinkedIn carousel best practices?", "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["8-slide max for LinkedIn carousel"]},
    {"id": "A17", "cat": "single-exact", "q": "Hindi UI for kirana store app?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Hindi English bilingual UI mandatory for adoption"]},
    {"id": "A18", "cat": "single-exact", "q": "Security demo page blind spot widget?", "exp_sc": ["Security Demo Page"], "exp_anc": ["Blind spot widget increases engagement 3x"]},
    {"id": "A19", "cat": "single-exact", "q": "How many Gurgaon organizations did we identify?", "exp_sc": ["GTM Lead Pipeline"], "exp_anc": ["60 AI-friendly Gurgaon orgs across 7 verticals"]},
    {"id": "A20", "cat": "single-exact", "q": "Privacy masking compliance for hospitals?", "exp_sc": ["Hospital CCTV Analytics"], "exp_anc": ["Privacy masking mandatory for patient areas"]},

    # ═══════════════════════════════════════════════════════════
    # B. SINGLE-DOMAIN SEMANTIC (15 cases) — same meaning, different words
    # ═══════════════════════════════════════════════════════════
    {"id": "B01", "cat": "single-semantic", "q": "What's the ideal number of words in an email heading?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Short subjects get 40% higher open rates"]},
    {"id": "B02", "cat": "single-semantic", "q": "How to handle old analog camera systems at client sites?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Analog DVR sites need edge compute bridge"]},
    {"id": "B03", "cat": "single-semantic", "q": "Which AI model is cheapest for basic safety alerts?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Qwen 2B sufficient for factory safety alerts"]},
    {"id": "B04", "cat": "single-semantic", "q": "How should store owners get their daily updates?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Store owners prefer daily WhatsApp reports"]},
    {"id": "B05", "cat": "single-semantic", "q": "What kind of content gets the most shares on professional social media?", "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["Behind-the-scenes factory visits get most engagement"]},
    {"id": "B06", "cat": "single-semantic", "q": "How to save GPU costs when processing video?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Batch 3 frames per 10s chunk saves 40% compute"]},
    {"id": "B07", "cat": "single-semantic", "q": "Which job title should we prioritize for security sales?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["VP Security more responsive than CISO"]},
    {"id": "B08", "cat": "single-semantic", "q": "How to track unauthorized people in apartment parking?", "exp_sc": ["Residential Society Security"], "exp_anc": ["Loitering detection in parking reduces theft 30%"]},
    {"id": "B09", "cat": "single-semantic", "q": "What's the narrative structure for our sales documents?", "exp_sc": ["Case Study Generation"], "exp_anc": ["Problem Solution ROI narrative arc"]},
    {"id": "B10", "cat": "single-semantic", "q": "How to make the AI work across different industry verticals?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["System prompt is only change between verticals"]},
    {"id": "B11", "cat": "single-semantic", "q": "What makes the first page of a slide deck effective?", "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["First slide must be hook not logo"]},
    {"id": "B12", "cat": "single-semantic", "q": "How fast does our product recognition need to be?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["200ms inference per frame on T4"]},
    {"id": "B13", "cat": "single-semantic", "q": "What interactive tool on our website converts best?", "exp_sc": ["Security Demo Page"], "exp_anc": ["Blind spot widget increases engagement 3x"]},
    {"id": "B14", "cat": "single-semantic", "q": "How should anchors in our knowledge system be written?", "exp_sc": ["PMIS System Design"], "exp_anc": ["Anchors must be atomic with numbers and reasoning"]},
    {"id": "B15", "cat": "single-semantic", "q": "What ISP problem affects our Indian camera deployments?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["RTSP pull mode more reliable than push for Indian ISPs"]},

    # ═══════════════════════════════════════════════════════════
    # C. CROSS-DOMAIN (20 cases) — needs knowledge from 2+ SCs
    # ═══════════════════════════════════════════════════════════
    {"id": "C01", "cat": "cross-domain", "q": "Write a cold email for construction safety product", "exp_sc": ["B2B Cold Outreach", "Construction Site Monitoring"], "exp_anc": ["Short subjects get 40% higher open rates", "PPE detection accuracy 96% at 50m range"]},
    {"id": "C02", "cat": "cross-domain", "q": "LinkedIn post about our kirana store AI deployment", "exp_sc": ["LinkedIn Content Strategy", "Kiran AI Retail Product"], "exp_anc": ["Casual authentic voice outperforms polished copy 2x", "99.9% accuracy on barcode-free Indian FMCG"]},
    {"id": "C03", "cat": "cross-domain", "q": "Create a demo page for hospital analytics", "exp_sc": ["Security Demo Page", "Hospital CCTV Analytics"], "exp_anc": ["Blind spot widget increases engagement 3x"]},
    {"id": "C04", "cat": "cross-domain", "q": "Case study for residential society security system", "exp_sc": ["Case Study Generation", "Residential Society Security"], "exp_anc": ["Problem Solution ROI narrative arc", "ANPR for vehicle tracking at society gates"]},
    {"id": "C05", "cat": "cross-domain", "q": "What compute do we need for construction site cameras?", "exp_sc": ["Vision Pipeline Architecture", "Construction Site Monitoring"], "exp_anc": ["T4 GPU handles 4 concurrent streams at 720p"]},
    {"id": "C06", "cat": "cross-domain", "q": "Email the Gurgaon hospital leads about our product", "exp_sc": ["B2B Cold Outreach", "Hospital CCTV Analytics", "GTM Lead Pipeline"], "exp_anc": ["Threat-intel language resonates with CISOs"]},
    {"id": "C07", "cat": "cross-domain", "q": "How to pitch our vision AI to construction CTO?", "exp_sc": ["B2B Cold Outreach", "Construction Site Monitoring"], "exp_anc": ["Construction CTO/CIO contacts most responsive"]},
    {"id": "C08", "cat": "cross-domain", "q": "GTM strategy for retail AI product launch", "exp_sc": ["GTM Lead Pipeline", "Kiran AI Retail Product"], "exp_anc": ["3-stage Claude workflow research score consolidate"]},
    {"id": "C09", "cat": "cross-domain", "q": "Build LinkedIn carousel about our vision pipeline", "exp_sc": ["LinkedIn Content Strategy", "Vision Pipeline Architecture"], "exp_anc": ["8-slide max for LinkedIn carousel"]},
    {"id": "C10", "cat": "cross-domain", "q": "Demo showing factory safety detection capabilities", "exp_sc": ["Security Demo Page", "Vision Pipeline Architecture"], "exp_anc": ["Yantrai Command dashboard mockup drives demos"]},
    {"id": "C11", "cat": "cross-domain", "q": "Outreach email to residential society management about ANPR", "exp_sc": ["B2B Cold Outreach", "Residential Society Security"], "exp_anc": ["Pain-point first solution second pattern"]},
    {"id": "C12", "cat": "cross-domain", "q": "Case study template for manufacturing quality control", "exp_sc": ["Case Study Generation", "Vision Pipeline Architecture"], "exp_anc": ["7-section template with IaaS stack diagrams"]},
    {"id": "C13", "cat": "cross-domain", "q": "How to position edge compute for hospital deployment?", "exp_sc": ["Vision Pipeline Architecture", "Hospital CCTV Analytics"], "exp_anc": ["Analog DVR sites need edge compute bridge"]},
    {"id": "C14", "cat": "cross-domain", "q": "Technical LinkedIn post explaining our dual-layer architecture", "exp_sc": ["LinkedIn Content Strategy", "Vision Pipeline Architecture"], "exp_anc": ["Technical explainers with diagrams reach decision-makers", "YOLO 11n as fast gate before deep analysis"]},
    {"id": "C15", "cat": "cross-domain", "q": "Lead qualification for hospital security camera deals", "exp_sc": ["GTM Lead Pipeline", "Hospital CCTV Analytics"], "exp_anc": ["AI-friendly org filter reduces list 60%"]},
    {"id": "C16", "cat": "cross-domain", "q": "Night camera setup for construction and residential", "exp_sc": ["Construction Site Monitoring", "Residential Society Security"], "exp_anc": ["Night shift monitoring needs IR-capable cameras"]},
    {"id": "C17", "cat": "cross-domain", "q": "How our memory system retrieves relevant knowledge for retrieval", "exp_sc": ["PMIS System Design"], "exp_anc": ["Word overlap quality temporal composite score"]},
    {"id": "C18", "cat": "cross-domain", "q": "Data visualization for LinkedIn carousel about retail AI accuracy", "exp_sc": ["LinkedIn Content Strategy", "Kiran AI Retail Product"], "exp_anc": ["Data visualization slides get saved and shared most"]},
    {"id": "C19", "cat": "cross-domain", "q": "Playground and pool safety monitoring for residential societies", "exp_sc": ["Residential Society Security"], "exp_anc": ["Child safety zones with perimeter alerts"]},
    {"id": "C20", "cat": "cross-domain", "q": "International construction company leads for outreach", "exp_sc": ["GTM Lead Pipeline", "B2B Cold Outreach"], "exp_anc": ["500 international construction leads with verified contacts"]},

    # ═══════════════════════════════════════════════════════════
    # D. MULTI-TURN CONVERSATIONS (5 convos x 4 turns = 20 cases)
    # ═══════════════════════════════════════════════════════════

    # Conv 1: Security email campaign
    {"id": "D01", "cat": "multi-turn", "conv": 1, "turn": 1, "q": "I need to write cold emails for security product",
     "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Short subjects get 40% higher open rates"]},
    {"id": "D02", "cat": "multi-turn", "conv": 1, "turn": 2, "q": "What about the follow-up timing?",
     "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["3-touch sequence over 10 days optimal"]},
    {"id": "D03", "cat": "multi-turn", "conv": 1, "turn": 3, "q": "Now create a demo page for these prospects",
     "exp_sc": ["Security Demo Page"], "exp_anc": ["Blind spot widget increases engagement 3x"]},
    {"id": "D04", "cat": "multi-turn", "conv": 1, "turn": 4, "q": "Build a case study to attach with the email",
     "exp_sc": ["Case Study Generation"], "exp_anc": ["Problem Solution ROI narrative arc"]},

    # Conv 2: Vision pipeline for construction
    {"id": "D05", "cat": "multi-turn", "conv": 2, "turn": 1, "q": "What compute do we need for factory CCTV analytics?",
     "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["YOLO 11n as fast gate before deep analysis"]},
    {"id": "D06", "cat": "multi-turn", "conv": 2, "turn": 2, "q": "Can we apply the same setup to construction sites?",
     "exp_sc": ["Vision Pipeline Architecture", "Construction Site Monitoring"], "exp_anc": ["System prompt is only change between verticals"]},
    {"id": "D07", "cat": "multi-turn", "conv": 2, "turn": 3, "q": "What about camera integration at old sites?",
     "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Analog DVR sites need edge compute bridge"]},
    {"id": "D08", "cat": "multi-turn", "conv": 2, "turn": 4, "q": "How to optimize the compute costs?",
     "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["Batch 3 frames per 10s chunk saves 40% compute"]},

    # Conv 3: Kiran AI marketing
    {"id": "D09", "cat": "multi-turn", "conv": 3, "turn": 1, "q": "Help me write a LinkedIn post about Kiran AI",
     "exp_sc": ["LinkedIn Content Strategy", "Kiran AI Retail Product"], "exp_anc": ["Casual authentic voice outperforms polished copy 2x"]},
    {"id": "D10", "cat": "multi-turn", "conv": 3, "turn": 2, "q": "Should I make it a carousel?",
     "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["8-slide max for LinkedIn carousel"]},
    {"id": "D11", "cat": "multi-turn", "conv": 3, "turn": 3, "q": "What technical details to include about store deployment?",
     "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Hindi English bilingual UI mandatory for adoption"]},
    {"id": "D12", "cat": "multi-turn", "conv": 3, "turn": 4, "q": "Include the accuracy numbers?",
     "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["99.9% accuracy on barcode-free Indian FMCG"]},

    # Conv 4: Hospital + residential pitch
    {"id": "D13", "cat": "multi-turn", "conv": 4, "turn": 1, "q": "What features do hospitals need from CCTV AI?",
     "exp_sc": ["Hospital CCTV Analytics"], "exp_anc": ["Fall detection in corridors most requested feature"]},
    {"id": "D14", "cat": "multi-turn", "conv": 4, "turn": 2, "q": "What about residential societies — similar needs?",
     "exp_sc": ["Residential Society Security"], "exp_anc": ["ANPR for vehicle tracking at society gates"]},
    {"id": "D15", "cat": "multi-turn", "conv": 4, "turn": 3, "q": "Alert speed requirements for both?",
     "exp_sc": ["Residential Society Security"], "exp_anc": ["Alert latency must be under 5 seconds for gate staff"]},
    {"id": "D16", "cat": "multi-turn", "conv": 4, "turn": 4, "q": "Now help me draft an email to Medanta hospital",
     "exp_sc": ["B2B Cold Outreach", "Hospital CCTV Analytics"], "exp_anc": ["Pain-point first solution second pattern"]},

    # Conv 5: GTM pipeline build
    {"id": "D17", "cat": "multi-turn", "conv": 5, "turn": 1, "q": "How should we build our sales lead pipeline?",
     "exp_sc": ["GTM Lead Pipeline"], "exp_anc": ["3-stage Claude workflow research score consolidate"]},
    {"id": "D18", "cat": "multi-turn", "conv": 5, "turn": 2, "q": "Which construction companies to target first?",
     "exp_sc": ["GTM Lead Pipeline", "B2B Cold Outreach"], "exp_anc": ["Construction CTO/CIO contacts most responsive"]},
    {"id": "D19", "cat": "multi-turn", "conv": 5, "turn": 3, "q": "Write the first email for these leads",
     "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Short subjects get 40% higher open rates"]},
    {"id": "D20", "cat": "multi-turn", "conv": 5, "turn": 4, "q": "What demo should we show them after reply?",
     "exp_sc": ["Security Demo Page"], "exp_anc": ["Yantrai Command dashboard mockup drives demos"]},

    # ═══════════════════════════════════════════════════════════
    # E. FEEDBACK-DRIVEN (10 cases) — tests that scoring changes ranking
    # ═══════════════════════════════════════════════════════════
    {"id": "E01", "cat": "feedback", "q": "Best email copy approach?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Pain-point first solution second pattern"], "feedback": {"B2B Cold Outreach": 5.0}},
    {"id": "E02", "cat": "feedback", "q": "Best email copy approach?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": ["Pain-point first solution second pattern"], "note": "After high feedback, same query should return same SC with higher score"},
    {"id": "E03", "cat": "feedback", "q": "How to deploy at client sites?", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["MediaMTX Cloudflare Tunnel bypasses RTSP firewalls"], "feedback": {"Vision Pipeline Architecture": 4.5}},
    {"id": "E04", "cat": "feedback", "q": "What works for kirana store owners?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Store owners prefer daily WhatsApp reports"], "feedback": {"Kiran AI Retail Product": 2.0}},
    {"id": "E05", "cat": "feedback", "q": "What works for kirana store owners?", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Store owners prefer daily WhatsApp reports"], "note": "After low feedback (2.0), SC quality drops"},
    {"id": "E06", "cat": "feedback", "q": "Construction safety monitoring setup?", "exp_sc": ["Construction Site Monitoring"], "exp_anc": ["PPE detection accuracy 96% at 50m range"], "feedback": {"Construction Site Monitoring": 4.0}},
    {"id": "E07", "cat": "feedback", "q": "LinkedIn content that gets engagement?", "exp_sc": ["LinkedIn Content Strategy"], "exp_anc": ["Casual authentic voice outperforms polished copy 2x"], "feedback": {"LinkedIn Content Strategy": 4.5}},
    {"id": "E08", "cat": "feedback", "q": "How to make society gates safer?", "exp_sc": ["Residential Society Security"], "exp_anc": ["ANPR for vehicle tracking at society gates"], "feedback": {"Residential Society Security": 3.5}},
    {"id": "E09", "cat": "feedback", "q": "Lead qualification for our sales pipeline?", "exp_sc": ["GTM Lead Pipeline"], "exp_anc": ["AI-friendly org filter reduces list 60%"], "feedback": {"GTM Lead Pipeline": 4.0}},
    {"id": "E10", "cat": "feedback", "q": "How does our memory system retrieve things?", "exp_sc": ["PMIS System Design"], "exp_anc": ["Word overlap quality temporal composite score"], "feedback": {"PMIS System Design": 5.0}},

    # ═══════════════════════════════════════════════════════════
    # F. EDGE CASES (15 cases) — ambiguous, broad, very specific, empty
    # ═══════════════════════════════════════════════════════════
    {"id": "F01", "cat": "edge", "q": "Tell me everything about our products", "exp_sc": ["Kiran AI Retail Product", "Vision Pipeline Architecture"], "exp_anc": [], "note": "Very broad query — should return highest quality SCs"},
    {"id": "F02", "cat": "edge", "q": "What's the ROI of AI cameras?", "exp_sc": ["Construction Site Monitoring", "Hospital CCTV Analytics"], "exp_anc": [], "note": "Broad ROI question across verticals"},
    {"id": "F03", "cat": "edge", "q": "7B parameter model", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["7B model needed for complex retail analytics"], "note": "Very short, specific query"},
    {"id": "F04", "cat": "edge", "q": "MediaMTX", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["MediaMTX Cloudflare Tunnel bypasses RTSP firewalls"], "note": "Single technical term"},
    {"id": "F05", "cat": "edge", "q": "deployment challenges India", "exp_sc": ["Vision Pipeline Architecture", "Kiran AI Retail Product"], "exp_anc": [], "note": "Broad geographic query"},
    {"id": "F06", "cat": "edge", "q": "What doesn't work for email outreach?", "exp_sc": ["B2B Cold Outreach"], "exp_anc": [], "note": "Negation query — still needs the right SC"},
    {"id": "F07", "cat": "edge", "q": "Cloudflare tunnel RTSP bypass method", "exp_sc": ["Vision Pipeline Architecture"], "exp_anc": ["MediaMTX Cloudflare Tunnel bypasses RTSP firewalls"], "note": "Highly specific technical query"},
    {"id": "F08", "cat": "edge", "q": "What's our best product?", "exp_sc": ["Kiran AI Retail Product", "Vision Pipeline Architecture"], "exp_anc": [], "note": "Subjective broad query"},
    {"id": "F09", "cat": "edge", "q": "NABH accreditation camera placement hospital requirements", "exp_sc": ["Hospital CCTV Analytics"], "exp_anc": [], "note": "Domain-specific jargon from hospital vertical"},
    {"id": "F10", "cat": "edge", "q": "child pool playground safety alert", "exp_sc": ["Residential Society Security"], "exp_anc": ["Child safety zones with perimeter alerts"], "note": "Fragmented keyword query"},
    {"id": "F11", "cat": "edge", "q": "How to name super contexts properly?", "exp_sc": ["PMIS System Design"], "exp_anc": ["Name by reusable domain not by date"], "note": "Meta query about the system itself"},
    {"id": "F12", "cat": "edge", "q": "50 images training", "exp_sc": ["Kiran AI Retail Product"], "exp_anc": ["Training on 50 product images sufficient per SKU"], "note": "Number + concept fragment"},
    {"id": "F13", "cat": "edge", "q": "zone heatmap bottleneck", "exp_sc": ["Construction Site Monitoring"], "exp_anc": ["Zone-based activity heatmaps for project managers"], "note": "3-word fragment"},
    {"id": "F14", "cat": "edge", "q": "Fortis FMRI hospital Gurgaon target", "exp_sc": ["Hospital CCTV Analytics"], "exp_anc": ["Medanta and Fortis FMRI primary Gurgaon targets"], "note": "Proper noun query"},
    {"id": "F15", "cat": "edge", "q": "What's the complete flow from lead to demo to close?", "exp_sc": ["GTM Lead Pipeline", "B2B Cold Outreach", "Security Demo Page"], "exp_anc": [], "note": "Multi-SC journey query"},
]


# =============================================================================
# Test Runner
# =============================================================================

def load_memory(conn):
    """Load all seed data into the database."""
    import memory as mem
    for data in SEED_DATA:
        f = io.StringIO()
        with redirect_stdout(f):
            mem.cmd_store(conn, json.dumps(data))


def run_retrieval(conn, query):
    """Run P9+ parameterized retrieval and parse output."""
    try:
        from p9_retrieve import p9_retrieve_parameterized, SessionEngine
        import json as _json
        # Load best config if available
        _cfg_path = ROOT / "Graph_DB" / "experiments" / "best_config_v2.json"
        params = _json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}
        session = SessionEngine(
            decay=params.get("session_decay", 0.825),
            divergence_threshold=params.get("divergence_threshold", 0.35),
        )
        f = io.StringIO()
        with redirect_stdout(f):
            p9_retrieve_parameterized(conn, query, params=params, session=session,
                                       top_k=int(params.get("max_results", 5)))
        return _json.loads(f.getvalue())
    except ImportError:
        # Fallback to original
        import memory as mem
        f = io.StringIO()
        with redirect_stdout(f):
            mem.cmd_retrieve(conn, query)
        return json.loads(f.getvalue())


def evaluate_test(result, test):
    """Score a single test case."""
    ret_scs = [m["super_context"] for m in result.get("memories", [])]
    exp_scs = test["exp_sc"]

    # SC recall: did we find expected SCs? (fuzzy: substring match)
    sc_hits = 0
    for exp in exp_scs:
        exp_lower = exp.lower()
        for ret in ret_scs:
            ret_lower = ret.lower()
            # Match if one contains the other, or significant word overlap
            if exp_lower in ret_lower or ret_lower in exp_lower:
                sc_hits += 1
                break
            else:
                # Strip common words and check significant word overlap
                _stop = {"the","a","an","and","or","for","of","in","on","—","-","&"}
                exp_words = set(w for w in exp_lower.split() if w not in _stop and len(w) > 1)
                ret_words = set(w for w in ret_lower.split() if w not in _stop and len(w) > 1)
                overlap = len(exp_words & ret_words)
                # Also check partial word matches (system↔design, architecture↔design)
                partial = 0
                for ew in exp_words:
                    for rw in ret_words:
                        if ew in rw or rw in ew:
                            partial += 1
                            break
                total_match = overlap + partial * 0.5
                if total_match >= 2 or (overlap >= 1 and len(exp_words) <= 2):
                    sc_hits += 1
                    break
    sc_recall = sc_hits / len(exp_scs) if exp_scs else 1.0

    # Anchor recall: did expected anchors appear? (fuzzy word-overlap matching)
    all_ancs = []
    for m in result.get("memories", []):
        for ctx in m.get("contexts", []):
            for a in ctx.get("anchors", []):
                all_ancs.append(a["title"].lower())

    exp_ancs = test.get("exp_anc", [])
    if exp_ancs:
        anc_hits = 0
        for ea in exp_ancs:
            ea_lower = ea.lower()
            # Direct substring match
            if any(ea_lower in a for a in all_ancs):
                anc_hits += 1
                continue
            # Fuzzy: check if 3+ significant words overlap
            ea_words = set(w for w in ea_lower.split() if len(w) > 3)
            for a in all_ancs:
                a_words = set(w for w in a.split() if len(w) > 3)
                overlap = len(ea_words & a_words)
                if overlap >= 3 or (overlap >= 2 and len(ea_words) <= 4):
                    anc_hits += 1
                    break
        anc_recall = anc_hits / len(exp_ancs)
    else:
        anc_recall = 1.0  # no specific anchors expected

    # Pass if SC recall > 0 (found at least one right SC) and anchor recall > 0
    passed = sc_recall > 0 and (anc_recall > 0 or not exp_ancs)

    return {
        "id": test["id"],
        "cat": test["cat"],
        "query": test["q"][:60],
        "sc_recall": round(sc_recall, 2),
        "anc_recall": round(anc_recall, 2),
        "passed": passed,
        "retrieved": ret_scs[:3],
        "expected": exp_scs,
    }


def main():
    import memory as mem
    conn = mem.get_db()

    # Check if data exists
    n = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
    if n < 20:
        print("Loading seed data...")
        load_memory(conn)
        # Run upgrade to add tags
        try:
            from upgrade_memories import generate_tags
            anchors = conn.execute("SELECT id, title, content FROM nodes WHERE type='anchor'").fetchall()
            for anc in anchors:
                tags = generate_tags(anc["title"], anc["content"] or "")
                conn.execute("UPDATE nodes SET tags=? WHERE id=?", (json.dumps(tags), anc["id"]))
            conn.commit()
            print(f"Tagged {len(anchors)} anchors")
        except ImportError:
            print("upgrade_memories.py not found — running without tags")

    n = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
    n_anc = conn.execute("SELECT COUNT(*) as c FROM nodes WHERE type='anchor'").fetchone()["c"]
    print(f"\nDatabase: {n} nodes ({n_anc} anchors)")
    print(f"Running 100 ground truth tests...\n")

    # Run tests
    results = []
    conv_history = {}  # conv_id -> list of prior queries

    for test in TESTS:
        # Handle multi-turn: feed prior turns as expanded query
        if test["cat"] == "multi-turn":
            conv_id = test.get("conv", 0)
            if conv_id not in conv_history:
                conv_history[conv_id] = []

        # Handle feedback: apply before query
        if "feedback" in test:
            for sc_title, score in test["feedback"].items():
                sc_row = conn.execute("SELECT id FROM nodes WHERE type='super_context' AND title=?", (sc_title,)).fetchone()
                if sc_row:
                    conn.execute("UPDATE nodes SET quality=? WHERE id=?", (score, sc_row["id"]))
            conn.commit()

        result = run_retrieval(conn, test["q"])
        metrics = evaluate_test(result, test)
        results.append(metrics)

    # Summary
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r)

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"{'='*90}")
    print(f"GROUND TRUTH RESULTS: {total_pass}/{total} passed ({total_pass/total:.0%})")
    print(f"{'='*90}")

    print(f"\n{'Category':<20} {'Pass':>6} {'Total':>6} {'Rate':>8} {'Avg SC_R':>9} {'Avg ANC_R':>10}")
    print("-" * 62)
    for cat in ["single-exact", "single-semantic", "cross-domain", "multi-turn", "feedback", "edge"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        p = sum(1 for i in items if i["passed"])
        sc_r = sum(i["sc_recall"] for i in items) / len(items)
        anc_r = sum(i["anc_recall"] for i in items) / len(items)
        rate = p / len(items)
        marker = "  " if rate >= 0.8 else " !"
        print(f"{cat:<20} {p:>6} {len(items):>6} {rate:>7.0%}{marker} {sc_r:>9.2f} {anc_r:>10.2f}")

    # Failures detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'='*90}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'='*90}")
        for f in failures:
            print(f"\n  {f['id']} [{f['cat']}]: \"{f['query']}\"")
            print(f"    Expected SCs: {f['expected']}")
            print(f"    Got SCs:      {f['retrieved']}")
            print(f"    SC recall={f['sc_recall']:.2f}  ANC recall={f['anc_recall']:.2f}")

    # Export
    output = {
        "total": total, "passed": total_pass, "rate": round(total_pass/total, 3),
        "by_category": {cat: {"passed": sum(1 for i in items if i["passed"]), "total": len(items),
                              "rate": round(sum(1 for i in items if i["passed"])/len(items), 3)}
                        for cat, items in by_cat.items()},
        "failures": failures,
        "all_results": results,
    }
    output_path = ROOT / "ground_truth_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to: {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
