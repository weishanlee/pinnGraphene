# paperAgent

A template repository for AI-assisted academic paper generation using Claude Code, designed for mathematical modeling competitions (MCM/ICM, HiMCM, IMMC) and peer-reviewed journal submissions. Now enhanced with breakthrough research analysis to transform papers from incremental to innovative, with mandatory workflow checklists and enhanced verification systems.

## Features

- **Adaptive Research-First Workflow (DEFAULT)**: Default workflow validates theorems before claims; traditional workflow available by request
- **Competition Support**: Templates for MCM/ICM, HiMCM, and IMMC competitions with format-specific rules
- **Journal Templates**: Pre-configured templates for Elsevier, Springer, AMCS, and ASCE journals
- **Mandatory Workflow Checklists**: Required use of `prompts_workflow_checklist` for EVERY section
- **Code Audit Protocol**: MANDATORY audit of existing v* codebase before creating ANY Python code to prevent duplicated implementations
- **Calculation Report Generation**: Scripts automatically generate `report_calculation_v*.md` for paper cross-referencing
- **Review Checklist Implementation**: MANDATORY dual-version checklists (comment + visible) with verification script to ensure visibility in PDFs
- **Direct File Editing Protocol**: Enforces direct editing of existing files, preventing creation of confusing intermediate files (compile_*.tex, temp_*.tex)
- **Research Breakthrough Analysis**: Deep gap identification (10+ gaps) and innovation development with quality gates
- **Title Generation**: AI-powered title suggestions with scoring criteria (clarity, innovation, accuracy, memorability)
- **Large Figure Collection Handling**: Systematic organization for 20-100+ figures with mandatory inclusion of ALL user-provided figures
- **Figure Verification Protocol**: Mandatory pre-compilation verification of all figure paths with automatic copying from timestamped directories
- **Code Execution Handoff**: Enhanced with mandatory bug-free validation, execution planning, and progress tracking
- **Enhanced Code Debugging**: MANDATORY 10-minute (600s) debug run before handoff with comprehensive validation
- **Test Result Display**: Clear visual indicators - '✓' for PASSED and '✗' for FAILED across all validation outputs
- **Code Execution Summaries**: Automatic documentation of all outputs after running scripts, including figures, data, models, and logs
- **Long-Running Script Monitoring**: Comprehensive monitoring system for scripts exceeding 10-minute execution limit with status tracking, error logging, and checkpoint recovery
- **Asynchronous Debugging**: Debug failed scripts using collected logs after execution timeout, with root cause analysis and recovery suggestions
- **Automated Workflows**: Streamlined paper generation with section-by-section compilation and mandatory review checkpoints
- **Web Scraping**: Automated research paper collection with multi-stage verification pipeline
- **Enhanced Verification System**: 
  - Forbidden pattern detection ("Unknown Authors", "Draft version", etc.)
  - Batch validation rules with automatic halt conditions
  - Circuit breakers (5 consecutive failures, >25% failure rate)
  - Suspicious pattern detection (identical abstracts, short titles, single-letter authors)
  - **Bibliography verification workflow** with Playwright MCP for every reference
  - **Fake reference replacement** workflow to maintain academic integrity
  - **Citation completeness check** with Playwright MCP to find missing metadata (journal names, volume, pages)
  - **NEW: BibTeX file download** for every paper with citation key renaming
- **Citation Balance Enforcement**: 
  - Mandatory alternation between author-prominent and information-prominent citation styles
  - Prevents monotonous citation patterns and improves readability
  - Automatic detection of dangling citations (e.g., "[1] demonstrates...")
  - Built into all review prompts for quality assurance
- **Data Analysis**: Built-in statistical analysis and visualization tools
- **Bibliography Management**: Automated citation extraction with section-specific bibliographies
- **Bibliography Version Synchronization**: MANDATORY version matching between tex files and bibliographies (methods_v11.tex → methods_refs_v11.bib) to prevent missing references
- **Bibliography-First Compilation Protocol**: Enforces bibliography creation BEFORE compilation to prevent [?] citations
- **Duplicate Reference Prevention**: Smart content-based deduplication across sections using DOI, title similarity (>85%), and author-year matching with 5% threshold kill switch
- **Literal Interpretation Protocol**: Enforces exact usage of user-provided information without additions, corrections, or assumptions
- **Caption Length Management**: Automatic detection and prevention of caption overflow (200 character limit)
- **PDF Naming Enforcement**: MANDATORY creation of both wrapper.pdf AND section.pdf files for proper user review
- **Figure Preservation Protocol**: Prevents unintended figure modifications during reviewer feedback implementation
- **Reviewer Response Protocol**: Systematic comment classification before implementation to prevent scope creep
- **Smart Version Detection**: MANDATORY verification protocol ensures highest version selected unless documented reason; prevents v2 vs v3 content loss
- **Introduction Section Method Synchronization**: MANDATORY use of LATEST methods_v*.tex when writing Introduction section to ensure accurate model/strategy descriptions
- **Content Integrity**: 
  - Sentence-level verification (EVERY sentence must cite specific sources)
  - Prohibited content enforcement (no fabricated data, nonsense, lies, or hallucinations)
  - Paragraph-by-paragraph final verification before compilation
- **Figure Validation**: Enhanced box overlap checking with professional visual quality
- **Diagram Requirements**: 
  - **Complexity Limits**: Maximum 12 components, 15 connections, 3 hierarchy levels per diagram
  - **Progressive Disclosure**: Automatic splitting of complex diagrams into overview + details
  - **Spacing Requirements**: Minimum 80px between elements, 40% whitespace ratio
  - **Font Size Minimums**: Title 32pt, Labels 20pt, Annotations 14pt for readability
  - **Pre-Generation Checklist**: Mandatory validation before creating any diagram
  - **Emergency Simplification**: Protocol for handling overly complex diagrams
  - Focus on visual style with icons, gradients, and domain-appropriate metaphors
- **LaTeX Package Management & Synchronization**: Automatic detection, tracking, and synchronization of LaTeX packages from section wrappers to main.tex to prevent compilation failures
- **Post-Compilation File Listing**: Displays and saves comprehensive list of all files used in final PDF
- **Journal Submission Documents**: MANDATORY generation of Cover Letter and Highlights in .odt format using Python with odfpy library (NO .txt files allowed)
- **Submit Folder Preparation**: MANDATORY Step 10 copies all final files to submit/ folder for direct submission (no submission from output/ folder)
- **Peer Review Support**: AI reviewer prompts for quality assurance (can be used by ChatGPT 5 Pro or Gemini 2.5 Pro externally, or Claude Code when explicitly requested)
- **Ultrathink Integration**: Enhanced complex task handling with "Ultrathink with step-by-step reasoning" in critical prompts for methods development, code debugging, statistical analysis, and innovation

## 🎯 DEFAULT: Adaptive Research-First Workflow

### Overview
The system now uses **Adaptive Research-First Workflow as the DEFAULT** to ensure research integrity and claims match actual achievements.

#### Adaptive Research-First Workflow (DEFAULT)
**Writing Order**: Problem Analysis → Methods → Code & Results → Introduction → Conclusions → Abstract/Summary

**This is now the default because it**:
- Ensures introduction matches actual results (not hopes)
- Validates theorems before claiming they work
- Prevents overpromising in research claims
- Allows methods to evolve based on discoveries
- Maintains research integrity throughout

#### Traditional Workflow (Alternative - By Request Only)
**Writing Order**: Introduction → Methods → Results → Conclusions → Abstract/Summary

**Only use when**:
- User explicitly requests traditional workflow
- Problem has well-established solution methods
- Only minor modifications to known algorithms
- Very clear path from problem to solution
- Severe time constraints

### Key Benefits of Adaptive Workflow

1. **Validated Claims**: Introduction accurately reflects what was ACTUALLY achieved, not what was hoped for
2. **Reduced Rework**: No need to rewrite introduction when methods don't work as expected
3. **Theorem-First Development**: Develop and validate theorems before making claims about them
4. **Honest Research**: Prevents overpromising in introduction by writing it after results are proven
5. **Iterative Refinement**: Methods can evolve based on code discoveries without conflicting with introduction

### How It Works

1. **Problem Analysis Phase** (Adaptive Only):
   - Analyze the problem using `prompts_problem_analysis`
   - Generate `output/problem_analysis.md` with solution strategies
   - Define success criteria before implementation

2. **Methods Development**:
   - Write methods speculatively with multiple approaches
   - Mark uncertain elements with `% TO VERIFY` comments
   - Create `methods_v1.tex` with theoretical approaches

3. **Code & Validation**:
   - Implement and test the methods
   - Discover what actually works
   - Update methods if code improvements found (methods_v2.tex)

4. **Introduction Writing** (After Validation):
   - Reference ACTUAL results from validated methods
   - Use past tense: "We developed...", "We demonstrated..."
   - Make claims backed by proven performance

5. **Final Assembly**:
   - All claims are validated
   - No misrepresentation of achievements
   - Coherent narrative based on actual results

### Workflow Selection

**IMPORTANT**: Adaptive workflow is now the DEFAULT for all projects.
- The system will automatically use the Adaptive workflow unless you explicitly request Traditional
- To use Traditional workflow, you must specifically ask for it
- `prompts_workflow_checklist` starts with Adaptive selected by default
- Both workflows are fully supported, but Adaptive is strongly recommended

### Implementation Details

**New Files Created**:
- `prompts/prompts_problem_analysis` - Internal problem analysis for adaptive workflow
- Modified `prompts/prompts_introduction` - Supports writing after validation
- Enhanced `prompts/prompts_methods` - Supports speculative writing with verification markers
- Updated workflow files with dual-path options

**Backward Compatibility**: 
The traditional workflow remains fully functional. Existing projects are unaffected.

## Project Structure

```
paperAgent_Template/
├── problem/            # Competition problems or research outlines (PDF)
├── data/              # User-provided data files
├── figures/           # User-provided figures
├── codes/             # User-provided implementations
├── papers/            # User-provided research papers
├── output/            # Generated materials (LaTeX, PDFs, figures, code, submission documents)
│   ├── papers/        # Downloaded research papers
│   │   ├── arxiv_papers/
│   │   ├── journal_papers/
│   │   └── paper_bib/  # Downloaded .bib files renamed with citation keys
│   └── review_reports/  # AI reviewer outputs (from ChatGPT 5 Pro, Gemini 2.5 Pro, or Claude Code reviews)
├── submit/            # Files ready for submission to arXiv and publishing on GitHub website
├── templates/         # LaTeX templates for competitions and journals
├── utilityScripts/    # Helper scripts for automation
│   ├── detect_latest_sections.py  # Smart version detection with content analysis
│   ├── extract_section_citations.py
│   ├── ensure_bibliography_sync.py # Automatic bibliography version synchronization (NEW)
│   ├── merge_bibliographies.py
│   ├── gap_analysis_formatter.py  # Creates gap analysis templates
│   ├── detect_required_packages.py  # LaTeX package dependency detection
│   ├── verify_bibliography_v1.py  # Verify references are genuine (NEW)
│   ├── replace_fake_references_v1.py  # Replace fake refs with genuine ones (NEW)
│   ├── monitoring_base.py         # Base class for monitored scripts
│   ├── check_status.py           # Check script execution status
│   ├── runner.py                 # Universal script runner with monitoring
│   └── monitor_wrapper.sh        # Bash wrapper for monitoring
├── prompts/           # Detailed prompts with concrete examples
│   ├── prompts_workflow_checklist    # MANDATORY checklist for EVERY section
│   ├── prompts_problem_analysis      # Problem analysis for adaptive workflow (NEW)
│   ├── prompts_section_compilation    # Bibliography extraction examples and commands
│   ├── prompts_clean_section_files   # Clean file generation for final assembly
│   ├── prompts_clean_file_environment_rules # Environment nesting prevention guide
│   ├── prompts_paper_content_verification  # Content verification examples
│   ├── prompts_review_checkpoint     # Section review checklist formats and examples
│   ├── prompts_figure_box_validation # Box layout validation code
│   ├── prompts_figure_verification   # Pre-compilation figure path verification (NEW)
│   ├── prompts_figure_preservation_protocol # Figure management during revisions (NEW)
│   ├── prompts_reviewer_response_protocol # Systematic feedback implementation (NEW)
│   ├── prompts_verification_failure_protocol # Failure handling examples
│   ├── prompts_version_selection_protocol # Version selection examples and implementation
│   ├── prompts_version_selection_mandatory_check # MANDATORY version verification steps (NEW)
│   ├── prompts_version_naming_rules  # Version naming rules and examples
│   ├── prompts_file_creation_checklist # Pre-file creation validation (NEW)
│   ├── prompts_direct_editing_protocol # Direct file editing enforcement (NEW)
│   ├── prompts_webScraping           # Web scraping verification examples
│   ├── prompts_final_assembly        # Final PDF assembly examples and commands
│   ├── prompts_research_breakthrough # Deep gap analysis and innovation development
│   ├── prompts_title_generation      # Title generation with scoring criteria
│   ├── prompts_code_execution_instructions # Long-running script handoff guidelines
│   ├── prompts_code_timeout_protocol # Timeout handling procedures (no timeout in scripts)
│   ├── prompts_execution_preflight_check # Pre-execution timeout detection
│   ├── prompts_code_execution_summary # Post-execution output documentation
│   ├── prompts_code_debugging_protocol # MANDATORY 10-min debug run procedures
│   ├── prompts_pre_handoff_validation # Pre-handoff validation checklist
│   ├── prompts_debugging_workflow # 600-second debugging requirements
│   ├── prompts_long_running_monitoring # Long-running script monitoring implementation
│   ├── prompts_asynchronous_debugging # Debug scripts using logs after timeout
│   ├── prompts_AIReport              # AI usage disclosure (MANDATORY)
│   ├── prompts_appendix              # General appendix instructions (OPTIONAL)
│   ├── prompts_cover_letter_highlights # Journal submission documents generation
│   ├── prompts_odt_generation_mandatory # MANDATORY ODT file generation with Python/odfpy
│   ├── prompts_submission_folder_mandatory # MANDATORY submit folder preparation instructions
│   ├── prompts_verification_testing  # Pre-flight tests for verification system
│   ├── prompts_writing_style_journal # Scientific narrative flow for journal papers (NEW)
│   ├── prompts_journal_paper_narrative_enforcement # Narrative flow enforcement for journal papers (NEW Aug 5)
│   ├── prompts_github_usage         # GitHub repository reference rules (NEW)
│   ├── prompts_citation_reference_validation # Citation/reference validation (NEW)
│   ├── prompts_bibliography_validation # Complete citation validation protocol (NEW)
│   ├── prompts_bibliography_verification_workflow # Verify references with Playwright (NEW)
│   ├── prompts_bibliography_replacement_workflow # Replace fake references (NEW)
│   ├── prompts_bibliography_version_sync # Version synchronization for bibliographies (NEW)
│   ├── prompts_bibliography_compilation_protocol # Step-by-step bibliography creation (NEW)
│   ├── prompts_bibliography_deduplication # Smart duplicate detection and removal (NEW)
│   ├── prompts_exact_information_mandatory # Use user-provided info EXACTLY (NEW)
│   ├── prompts_figure_preservation_strict # Strict figure preservation during revisions (NEW)
│   ├── prompts_version_modification_protocol # Surgical precision for file modifications (NEW)
│   ├── prompts_bibliography_entry_strict # Create bib entries with ONLY provided info (NEW)
│   ├── prompts_clean_rebuild_protocol # Rebuild from clean with minimal changes (NEW)
│   ├── prompts_pdf_naming_mandatory # Enforces both wrapper and user PDFs (NEW)
│   ├── prompts_caption_length_validation # Caption overflow prevention (NEW)
│   ├── prompts_statistical_verification # Statistical claims verification workflows (NEW)
│   ├── prompts_code_before_text     # Code-First protocol for revisions (NEW)
│   ├── prompts_code_audit_mandatory # Mandatory code audit procedures (NEW)
│   ├── prompts_extend_not_recreate  # Extension patterns and decision matrix (NEW)
│   ├── prompts_version_inheritance  # Version progression rules (NEW)
│   ├── prompts_mcp_vs_python_clarification # MCP vs Python distinction guide (NEW)
│   └── prompts_Reviewer/  # Peer review prompts for ChatGPT 5 Pro and Gemini 2.5 Pro (for reviewing already-generated content)
│       ├── prompts_review_abstract   # Abstract evaluation
│       ├── prompts_review_introduction # Introduction assessment
│       ├── prompts_review_methods    # Methods scrutiny
│       ├── prompts_review_resultsDisussions # Results/Discussion analysis
│       ├── prompts_review_conclusions # Conclusions review
│       ├── prompts_review_reference  # Bibliography quality check (with Playwright MCP)
│       ├── prompts_review_bibliography_validation # Deep reference validation (NEW)
│       ├── prompts_review_main       # Holistic manuscript evaluation including main.tex
│       ├── prompts_making_corrections_single_section # Section revision support
│       └── prompts_making_corrections_entire_manuscript # Full revision with response letter
└── CLAUDE.md          # Streamlined instructions for Claude Code (~39.4k chars, under 40k limit)
```

## User Manual

For detailed usage instructions and workflows, please refer to **[USER_MANUAL.md](USER_MANUAL.md)** which contains:
- Quick Start guide
- Writing papers from scratch
- Section-specific writing instructions
- Working with user-provided materials
- Code debugging workflows with ChatGPT 5 Pro and Gemini
- Peer review and revision processes  
- Finalization procedures

## 📚 Bibliography Verification & Replacement

### Overview

The bibliography verification and replacement system ensures academic integrity by verifying every reference using automated tools and Playwright MCP. This system is integrated at critical points throughout the paper generation workflow.

### CRITICAL: Bibliography Version Synchronization

**⚠️ MANDATORY**: Bibliography files MUST match tex file versions EXACTLY to prevent missing references.

#### Version Sync Rules:
- `methods_v11.tex` → `methods_refs_v11.bib` (NOT methods_refs.bib)
- Wrapper must reference same version: `\bibliography{methods_refs_v11}`
- ALWAYS extract new bibliography for each version

#### Automatic Version Sync:
```bash
# Before ANY compilation, ensure version sync:
cd output
python ../utilityScripts/ensure_bibliography_sync.py methods_v11.tex

# This will:
# 1. Check if methods_refs_v11.bib exists
# 2. If not, extract it from methods_v11.tex
# 3. Confirm wrapper should use \bibliography{methods_refs_v11}
```

**Common Failures and Prevention**:
- ❌ Using `methods_refs.bib` for `methods_v11.tex` → Missing bibliography in PDF
- ❌ Wrapper references `methods_refs_v1` for `methods_v11.tex` → Citations show as [?]
- ✅ CORRECT: All version numbers match exactly

### MANDATORY: Thorough Bibliography Check After Every Section

**⚠️ CRITICAL**: After compiling ANY section PDF, you MUST verify all references are genuine. No section is complete until bibliography passes verification.

#### Automated Verification
```bash
# Run verification script after each section compilation
cd output
~/.venv/ml_31123121/bin/python ../utilityScripts/verify_bibliography_v1.py \
  --section introduction --version 1

# Check the report
cat bibliography_verification_report_v1.md
```

**Features of verify_bibliography_v1.py**:
- Checks for downloaded .bib files in `output/papers/paper_bib/`
- Detects forbidden patterns (Unknown Authors, etc.)
- Validates metadata completeness
- Generates detailed verification reports
- Implements kill switches (>20% problematic = STOP)

#### Manual Verification with Playwright MCP
If automated verification finds issues, use Playwright MCP:
```
# For each suspicious reference, verify with:
playwright.search('"{exact title}" site:scholar.google.com')
playwright.search('{first_author} {year} "{title}"')
playwright.search('"{journal name}" academic journal')
```

### Replace Fake References with Genuine Ones

When fake/incomplete references are detected:

#### Automated Replacement
```bash
# Run replacement script
cd output
~/.venv/ml_31123121/bin/python ../utilityScripts/replace_fake_references_v1.py \
  --section introduction --version 1 \
  --verification-report bibliography_verification_report_v1.md

# This will:
# 1. Analyze citation context in .tex file
# 2. Generate search suggestions for replacements
# 3. Create new versions (v2) of .tex and .bib files
# 4. Generate replacement report
```

**Features of replace_fake_references_v1.py**:
- Analyzes citation context in .tex files
- Generates search suggestions for replacements
- Creates new versions of .tex and .bib files
- Documents all replacements
- Maintains argumentative flow

#### Manual Replacement Process
1. **Analyze Context**: Find WHERE the fake reference is cited in the .tex file
2. **Search for Replacements**: Use Playwright MCP to find genuine papers matching the context
3. **Validate Candidates**: Ensure replacements support the same claims
4. **Update Files**: Create `section_v2.tex` and `section_refs_v2.bib`
5. **Recompile**: Generate new PDF with corrected references
6. **Re-verify**: Run verification again on v2

### Workflow Integration

The verification system integrates at these critical points:
1. After section PDF compilation (workflow checklist item 6)
2. Before proceeding to next section (workflow checklist item 7)
3. During final assembly (final_check prompt)
4. In peer review process (review_bibliography_validation prompt)

### Kill Switch Conditions

**STOP IMMEDIATELY if**:
- >20% of references are fake/suspicious
- >10% have "Unknown Authors"
- >30% have incomplete metadata
- Same fake pattern appears 3+ times

### Verification Report Format

Each verification generates `bibliography_verification_report_v{n}.md`:
```markdown
## Summary
- ✅ Verified Genuine: 85 (89.5%)
- ❌ Fake/Suspicious: 5 (5.3%)
- ⚠️ Incomplete Metadata: 5 (5.3%)
- 📄 With Downloaded BibTeX: 78 (82.1%)

Status: NEEDS_REPLACEMENT
Action: Execute replacement workflow before proceeding
```

### Key Principles

1. **Quality over Quantity**: Better to have 80 genuine citations than 100 with 20 fake
2. **Verification is MANDATORY**: No section is complete until bibliography passes
3. **Downloaded BibTeX Priority**: Always check paper_bib/ folder first
4. **Context Preservation**: Replacements must support the same claims
5. **Version Control**: Always create new versions when making changes

### Success Metrics

A successful implementation will:
- Detect >95% of fake references
- Provide genuine replacements for all detected fakes
- Maintain citation context and paper flow
- Generate comprehensive documentation
- Prevent compilation with problematic bibliographies

### User Impact

This system ensures:
- Academic integrity through genuine citations
- Reduced risk of paper rejection
- Improved citation quality
- Clear documentation of all changes
- Confidence in bibliography authenticity


## Critical Format Differences

### Competition Papers (MCM/ICM, HiMCM, IMMC)
- **Introduction**: MUST have subsections (e.g., `\subsection{Problem Background}`)
- **Author Information**: NEVER include (anonymous submission)
- **Citation Style**: IEEE numeric style `[1], [2], [3]`
- **Required Sections**: Summary, Letter, Introduction, Methods, Results, Conclusions, AI Report
- **Page Limits**: Strict limits (check competition rules)

### Journal Papers
- **Introduction**: NO subsections - write as continuous narrative with NO bullet points
- **Author Information**: Include corresponding author and affiliations
- **Citation Style**: Journal-specific (check template)
- **Writing Style**: Scientific narrative prose - maximum 2 bullet points per page
- **Abstract/Conclusions**: MUST be pure narrative - NO bullet points allowed
- **Required Sections**: Abstract, Introduction, Methods, Results, Conclusions, AI Report
- **Submission Documents**: Cover Letter and Highlights in .odt format

## Available Templates

### Competition Templates
- MCM/ICM (Mathematical Contest in Modeling)
- HiMCM (High School Mathematical Contest in Modeling)
- IMMC (International Mathematical Modeling Challenge)

### Journal Templates
- Elsevier journals
- Springer journals
- Annals of Mathematics and Computer Science (AMCS)
- American Society of Civil Engineers (ASCE)

## Key Features

### ASCII-Only Requirement for arXiv Submission (NEW July 2025)
- **Abstracts and summaries MUST contain ONLY ASCII characters** to prevent arXiv rejection
- Common forbidden characters: em dash (—), multiplication sign (×), smart quotes (" ")
- Comprehensive list of forbidden characters and ASCII alternatives in `prompts_summaryOrAbstract`
- Automatic verification process to detect and replace non-ASCII characters
- Prevents "Bad character(s)" error during arXiv submission

### Performance Optimization (VERIFIED August 2025)
- **CLAUDE.md optimized to 39,971 chars** (under 40k limit) for optimal performance
- **Enhanced code debugging requirements**: 10-minute (600s) debug run before handoff, no timeout in final scripts
- **Verified consistency**: All prompts files are internally consistent with no conflicts
- **Verified references**: All prompt file references in CLAUDE.md are valid and exist
- Concrete examples relocated to specialized prompt files:
  - Code execution examples → `prompts_code_execution_instructions`
  - Web scraping verification → `prompts_webScraping`
  - Paper content verification → `prompts_paper_content_verification`
  - Section compilation examples → `prompts_section_compilation`
  - Final assembly examples → `prompts_final_assembly`
  - Citation method examples → `prompts_citation_methods`
  - GitHub usage examples → `prompts_github_usage`
  - Appendix content examples → `prompts_AIReport` and `prompts_appendix`
- CLAUDE.md now serves as streamlined index with references to detailed prompts
- Enhanced with mandatory workflow checklists for quality assurance
- Examples moved to prompts files to maintain performance while keeping essential content

### File Versioning System (ENHANCED July 2025)
- **MANDATORY sequential numbering for ALL files**: `introduction_v1.tex`, `introduction_v2.tex`, etc.
- **STRICT WORKFLOW**: When modifying ANY file, first check for previous versions, then use next sequential number
- **First file creation**: Always use `_v1` suffix (e.g., `analysis_v1.py`, `figure_v1.png`)
- **Any modification**: Create new file with incremented version (`_v2`, `_v3`...)
- **Applies to ALL file types**: LaTeX (.tex), Python (.py), figures (PNG/JPG/PDF), data files, markdown files
- **Helper/test files**: Include version (`utils_v1.py`, `test_v1.py`, `wrapper_v1.tex`)
- **Peer review files**: Include tex version (`introduction_peer_review_v2.md` for `introduction_v2.tex`)
- **STRICTLY FORBIDDEN naming patterns**: `_final`, `_updated`, `_revised`, `_new`, `_modified`, `_clean`
- Never overwrites existing versioned files - always creates new version
- See `prompts/prompts_version_naming_rules` for detailed examples and concrete workflows

### Methods-Code Bidirectional Workflow (ENHANCED August 2, 2025)
- **REVOLUTIONARY**: Methods and code evolve together through iterative improvement
- **Initial Implementation**: Code follows methods.tex exactly as before
- **Discovery Phase**: When code experiments yield >10% improvement, create NEW methods_v{n+1}.tex
- **Theory Evolution**: Document theoretical basis for improvements in updated methods version
- **Consistency Maintained**: Both code and theory remain synchronized throughout evolution
- **NEW**: Methods must develop theorems that explicitly address gaps from Introduction critique
- **NEW**: All mathematical foundations require proper citations from verified papers
- **NEW**: Mandatory visual representations (workflow, architecture, mechanism diagrams)
- **NEW**: Sensitivity analysis mandatory for competitions, justified analysis for journals
- **ENHANCED**: Bidirectional tracking ensures discoveries drive theoretical advances
- **ENHANCED**: Code comments document both current alignment and discovered improvements
- **ENHANCED**: See `prompts_methods_code_bidirectional` for comprehensive workflow details

### Incremental Compilation

### File Types in the Incremental Workflow

The incremental compilation system uses three types of files for each section:

#### 1. **Section .tex files** (e.g., `introduction_v1.tex`)
- **Contains**: Section content + review checklists/questions
- **NO bibliography commands**
- **Purpose**: Main content file that can be edited and versioned
- **Example structure**:
```latex
\section{Introduction}
[Section content here...]

% Review Checklist:
% [ ] Is the literature review comprehensive?
% [ ] Are citations properly formatted?
% Questions for user:
% 1. Should we expand on the theoretical framework?
```

#### 2. **Section wrapper files** (e.g., `introduction_v1_wrapper.tex`)
- **Contains**: Complete LaTeX document structure
- **Includes section .tex via `\input{}`**
- **Contains bibliography commands**
- **Purpose**: Enables standalone PDF compilation for review
- **Example structure**:
```latex
\documentclass{article}
\usepackage{cite}
\begin{document}
\input{introduction_v1.tex}  % Includes section content
\bibliography{introduction_refs_v1}  % Section-specific bibliography
\bibliographystyle{IEEEtran}
\end{document}
```

#### 3. **Section clean .tex files** (e.g., `introduction_clean.tex`)
- **Contains**: Section content ONLY (checklists removed)
- **NO bibliography commands**
- **NO review checklists/questions**
- **Purpose**: Production-ready content for final assembly
- **Created from**: Final approved version file

### Workflow Summary
```
Writing Phase:
introduction_v1.tex (content + checklist) 
    → introduction_v1_wrapper.tex (adds structure + bibliography) 
    → introduction_v1.pdf (for review)

Final Assembly:
introduction_v1.tex → introduction_clean.tex (removes checklist)
    → main.tex (includes all clean files + final bibliography)
    → main.pdf (final document)
```

### Key Points
- **Section files**: Reusable content with review aids
- **Wrappers**: Temporary compilation structures (not in final document)
- **Clean files**: Polished content for final PDF
- **Bibliographies**: In wrappers for review, in main.tex for final

### Additional Features
- Section-by-section PDF generation
- Review checkpoints after each section (checklists kept in .tex files but excluded from final PDF via clean files or markers)
- Reference Verification Summaries included in review PDFs but automatically excluded from final PDF
- Bibliography management per section
- Final merged bibliography named `ref_final.bib` (no version number as it's auto-generated)
- Single-column format for all PDFs

### Smart Version Selection with Content Analysis (NEW)
- **Content-aware selection**: Analyzes content completeness, not just modification time
- **Prevents content loss**: Warns when newer versions have fewer citations or missing content
- **Scoring system**: Evaluates citations, figures, tables, equations, and file size
- **Decision logging**: Documents why each version was selected
- **Manual override support**: Allows choosing specific versions when needed
- Handles multiple revisions (e.g., `methods.tex`, `methods_modified.tex`, `methods_v3.tex`)
- Generates detailed reports with warnings and recommendations

### Research Paper Collection
- Automated web scraping with verification
- Minimum 80 papers (50 arXiv + 30 peer-reviewed)
- Content verification and relevance checking
- PDF storage with metadata

### Data Analysis Tools
- Comprehensive data cleaning and validation
- 16+ visualization types
- Statistical hypothesis testing
- PCA, t-SNE, and advanced analytics

### Research Breakthrough Analysis (NEW)
- Deep literature gap identification (10+ gaps required)
- Innovation development strategies (theory synthesis, algorithm fusion)
- Breakthrough validation with novelty scoring
- Quality gates ensure genuine innovation (not incremental)
- Expected 15-30% performance improvements

### Quality Assurance
- **Mandatory workflow checklists** for every section (via `prompts_workflow_checklist`)
- Automated verification protocols with multi-stage pipeline
- Citation validation with strict "Unknown Authors" prohibition
- **Mathematical hallucination prevention** - Every formula must have complete derivation or proper citation
- **Forbidden pattern detection**:
  - Author patterns: "Unknown Author", "Anonymous", "N/A", "[Author Name]"
  - Title patterns: "Draft version", "arXiv:", "Untitled", "[Title]"
- **Batch validation rules**:
  - >10% unknown authors → STOP
  - >20% similar titles → STOP
  - >30% domain failures → STOP
  - >15% empty abstracts → STOP
- **Circuit breakers**:
  - 5 consecutive "Unknown Authors" → STOP
  - 10 consecutive verification failures → STOP
  - Same error pattern 3+ times → Switch method
  - Total failure rate >25% → Abort
- Figure validation: 10pt minimum font size, 30% horizontal/40% vertical spacing
- Pre-compilation checklists with comprehensive master verification
- Innovation validation metrics (novelty score ≥6/10, addresses 3+ gaps)
- **Sentence-level verification** - Every sentence must cite specific sources
- **Content truthfulness checks** - No fabricated data, nonsense, lies, or hallucinations
- **Kill switch protocol** - Stop → Fix → Recompile if any fabrication detected
- **Mandatory section summary verification** - Review all .tex files before final compilation

### Appendix Management (CRITICAL DISTINCTION)
- **AI Report Appendix (`appendixAIReport.tex`)**: 
  - MANDATORY for all papers (competitions and journals)
  - Contains AI tool usage disclosure for ethics/compliance
  - Lists specific queries and how AI was used
  - Required by competition rules and journal policies
- **General Appendix (`appendix.tex`)**:
  - OPTIONAL supplementary material
  - Contains extended results, proofs, extra figures/tables
  - Only include if you have overflow content from main sections
  - NOT for AI usage disclosure

### Enhanced Prompt System
- **prompts_workflow_checklist**: MANDATORY master checklist for EVERY section (UPDATED Aug 5 - enhanced journal paper narrative compliance)
- **prompts_workflow_competition**: Competition-specific workflow with format rules
- **prompts_workflow_journal**: Journal-specific workflow with format rules  
- **prompts_journal_paper_narrative_enforcement**: Narrative flow enforcement for journal papers (NEW Aug 5)
- **prompts_section_compilation**: Complete bibliography extraction examples, minimal compilation templates, procedures with versioned .bib files, and explicit PDF renaming instructions (UPDATED Aug 4)
- **prompts_clean_section_files**: Automated clean file generation removing review checklists for final assembly
- **prompts_paper_content_verification**: Multi-method extraction and automated recovery procedures
- **prompts_review_checkpoint**: Section review checklist formats (LaTeX and PDF) with complete examples
- **prompts_figure_box_validation**: Box layout validation implementation with strict spacing rules
- **prompts_verification_failure_protocol**: Pattern detection and circuit breaker implementations
- **prompts_version_selection_protocol**: Content-based version selection examples and implementation
- **prompts_webScraping**: Multi-stage verification pipeline with forbidden patterns (NOW includes concrete examples)
- **prompts_final_assembly**: Final PDF assembly with clean files, ref_final.bib merging, title generation, mandatory section summary, and post-compilation file listing
- **prompts_package_requirements**: LaTeX package detection and management instructions
- **prompts_research_breakthrough**: Deep gap analysis (10+ gaps) and innovation development
- **prompts_title_generation**: Title suggestions with scoring criteria
- **prompts_code_execution_instructions**: Enhanced with validation scripts, execution planning, progress tracking, concrete implementation examples, and output file naming guidelines (UPDATED Aug 2)
- **prompts_code_timeout_protocol**: Comprehensive timeout handling procedures - "timeout is a signal, not a problem"
- **prompts_methods**: Enhanced with mandatory bug-free validation checklist and execution planning (UPDATED)
- **prompts_package_requirements**: Complete LaTeX package detection with `detect_required_packages.py` implementation (UPDATED)
- **prompts_figure_box_validation**: Full implementations of `box_overlap_checker.py` and `smart_box_layout.py` (UPDATED)
- **prompts_introduction/results/conclusions**: Section-specific requirements with verification and package documentation
- **prompts_resultsAndDiscussions**: Distinguished competition vs journal paper figure organization (UPDATED Aug 5)
- **prompts_final_check**: Master verification with bibliography sanity check
- **prompts_cover_letter_highlights**: Journal submission documents generation with MANDATORY .odt format using Python/odfpy
- **prompts_odt_generation_mandatory**: Complete implementation for generating .odt files (NOT .txt) for journal submissions
- **prompts_submission_folder_mandatory**: Step-by-step submit folder preparation with automated scripts
- **prompts_verification_testing**: Pre-flight testing protocol for verification system
- **prompts_Reviewer**: Peer review prompts for manuscript quality assurance (usable by ChatGPT 5 Pro, Gemini 2.5 Pro, or Claude Code when explicitly requested)
- **prompts_user_provided_materials**: Guidelines for handling user-provided files in root folders (NEW)

### Working with User-Provided Materials

When users provide materials in root folders (`codes/`, `data/`, `figures/`, `papers/`), the system will:

1. **Automatically detect and copy** all user-provided files to corresponding `output/` subfolders
2. **Skip code execution steps** unless explicitly requested to verify or debug issues
3. **Use existing materials** as the foundation for writing the assigned section
4. **Preserve version control** by creating new versions (v2, v3...) if modifications are needed

The workflow adapts based on what materials are provided:
- **Codes provided**: Skip code development, analyze existing implementations
- **Data provided**: Skip data collection, use existing datasets for analysis
- **Figures provided**: Reference existing visualizations in the paper
- **Papers provided**: Include in literature review and bibliography

## Requirements

- Python virtual environments for web scraping and ML
- LaTeX distribution (TeXLive or MiKTeX)
- Claude Code CLI tool
- MCP tools (Playwright, Context7)

## Getting Started

1. Clone this repository
2. Set up the required virtual environments
3. Place your problem/outline in the `problem/` folder
4. Run Claude Code and follow the guided workflow


### Research Breakthrough Workflow

After web scraping but BEFORE writing any sections, the system performs deep gap analysis:

```bash
# System creates these files in output/
research_gaps_analysis.md      # Documents 10+ research limitations
breakthrough_proposal.md       # Your innovative approach
innovation_check.json         # Validation metrics

# Quality gates ensure:
- Novelty score ≥ 6/10 (7/10 for journals)
- Addresses 3+ major research gaps
- Quantifiable improvements (15-30%)
- Clear differentiation from existing work

# DO NOT proceed until innovation is validated!
```

### Gap Analysis Templates

Generate templates for breakthrough documentation:

```bash
cd output
~/.venv/ml_31123121/bin/python ../utilityScripts/gap_analysis_formatter.py --create-templates
```

### Title Generation

AFTER all sections are approved but BEFORE final PDF assembly:

```bash
# Prerequisites:
- All section PDFs approved
- Research breakthrough documented
- Results validated

# System generates output/title_suggestions.md with 5-7 options

# Each title scored on:
- Clarity score (problem clear?)
- Innovation score (breakthrough evident?)
- Accuracy score (matches content?)
- Memorability score (distinctive?)

# Example competition title:
"Hybrid Neural-Evolutionary Optimization for Urban Traffic Flow: A Multi-Scale Approach"

# Example journal title:
"Adaptive Hyperbolic Embeddings: A Novel Framework for Large-Scale Graph Analysis with Linear Complexity"
```

### Final PDF Assembly with Smart Version Selection

When assembling the final PDF, the system enforces MANDATORY version verification to prevent content loss:

#### 🚨 NEW: Mandatory Version Verification Protocol
Before ANY final assembly, the system now requires:
1. **Enumerate ALL versions** of each section file
2. **Compare content metrics** (size, citations, figures)
3. **Select HIGHEST version** unless documented reason exists
4. **Document decisions** in assembly_decisions.log

```bash
cd output

# MANDATORY FIRST: List all versions by section
for section in introduction methods resultsAndDiscussions conclusions; do
    echo "[$section versions]"
    ls -la ${section}*.tex 2>/dev/null | grep -v "_wrapper\|_clean"
done

# Run content analysis (REQUIRED)
~/.venv/ml_31123121/bin/python ../utilityScripts/detect_latest_sections.py --analyze-content

# This will:
# - Count citations, figures, tables in each version
# - Check for content loss between versions
# - Score each version based on completeness
# - Generate warnings if newer versions have less content
# - DEFAULT to highest version number (v3 > v2 > v1)

# Review the analysis report
cat section_detection_report.json
cat assembly_decisions.log  # Explains WHY each version was selected

# Common issues now prevented:
# - resultsAndDiscussions v2 selected when v3 exists (v3 often has more results)
# - methods v1 selected when v10+ exists (later versions have refined algorithms)
# - Content loss from selecting by modification time instead of version number
```

### 🛑 MANDATORY: Section Summary Review

After version detection, the system creates a comprehensive summary of all section files:

```bash
# System automatically creates final_sections_summary.md
cat final_sections_summary.md

# This summary includes:
# - All .tex files that will be used in main.pdf
# - Version numbers and content metrics for each section
# - Bibliography file counts
# - Supporting file counts (figures, data, code)

# CRITICAL: Review this summary BEFORE final compilation!
# The system will STOP and wait for your approval
# DO NOT proceed without user confirmation!
```

**Content Analysis Features**:
- Detects when revisions accidentally delete content
- Warns about missing citations, figures, or equations
- Prioritizes content completeness over modification time
- Supports manual override when automatic selection is wrong
- **NEW**: Creates mandatory section summary for review before compilation

### 📄 Post-Compilation File Listing

After successfully compiling main.pdf, the system automatically generates and displays a comprehensive file listing:

```bash
# System generates main_compilation_summary.md with:
# - All LaTeX section files used (with sizes and citation counts)
# - Bibliography file details
# - Figure files referenced
# - Summary statistics (total sections, citations, figures, PDF size/pages)

# The listing is:
# 1. Generated immediately after successful PDF compilation
# 2. Displayed on screen for user review
# 3. Saved to main_compilation_summary.md for reference
# 4. Presented BEFORE closing the compilation process

# Example output:
===============================================
POST-COMPILATION FILE LISTING:
===============================================
# Main.pdf Compilation Summary
Generated on: Fri Jul 17 2025
PDF: main.pdf (2.3M)

## LaTeX Section Files Used:
- introduction_v3.tex: 45K, 892 lines, 78 citations
- methods_revised.tex: 62K, 1243 lines, 95 citations
- results_final.tex: 38K, 756 lines, 42 citations
...
===============================================
```

This feature ensures complete transparency about which files were actually used in the final PDF compilation.

### 📨 Journal Submission Documents (MANDATORY .odt Format)

For journal papers only, the system generates professional submission documents after main.pdf compilation:

```bash
# ⚠️ CRITICAL: MUST use Python with odfpy library to generate .odt files
# ❌ NEVER create .txt files and suggest manual conversion
# ✅ See prompts/prompts_odt_generation_mandatory for implementation

# After main.pdf is compiled, the system:
# 1. Uses Playwright to research journal-specific requirements
# 2. Reviews all paper sections to extract key information
# 3. Generates two essential documents IN .ODT FORMAT:

# Cover Letter (cover_letter.odt):
# - Generated using Python with odfpy library
# - Addresses journal editor with paper title
# - Summarizes research importance and novelty
# - Highlights key merits and contributions
# - Explains fit with journal scope
# - Includes required declarations

# Highlights (highlights.odt):
# - Generated using Python with odfpy library
# - 5-6 bullet points (max)
# - Each point ≤85 characters
# - Present tense, self-contained
# - Captures research essence

# Example workflow:
===============================================
RESEARCHING JOURNAL REQUIREMENTS...
===============================================
Searching "Nature Methods cover letter requirements"...
Found: Cover letter should emphasize novelty, 1-2 pages
Found: Highlights not required for this journal

Generating cover_letter.odt...
✓ Cover letter created with 4 key contributions
✓ Validated against journal requirements
===============================================
```

**Key Features**:
- Journal-specific requirement research via web search
- Content extraction from approved sections
- Professional formatting in .odt format
- Automatic compliance validation
- Only for journal submissions (not competitions)

### 🚨 MANDATORY: Submit Folder Preparation

**CRITICAL**: After ALL documents are generated (main.pdf, cover_letter.odt, highlights.odt), files MUST be copied to the submit/ folder for submission.

#### Why Submit Folder is Required

The `submit/` folder serves as the final staging area for submission:
- **Located at project root** (not in output/)
- **Contains ONLY files ready for direct submission**
- **Self-contained** - can be zipped and sent as-is
- **Clean versions** - no review checklists or working files

#### What Gets Copied

The system executes Step 10 from `prompts_final_assembly` to copy:
1. **Clean section files** (abstract_clean.tex, introduction_clean.tex, etc.)
2. **Main documents** (main.tex, main.pdf, ref_final.bib)
3. **All figures** to submit/figures/ (from multiple source locations)
4. **Journal documents** (cover_letter.odt, highlights.odt)
5. **LaTeX class files** (if custom templates used)

#### Automated Submission Preparation

```bash
# After main.pdf and journal documents are created:
cd output
bash ../prepare_submission_v1.sh

# This script automatically:
# - Copies all clean .tex files
# - Copies main.pdf and bibliography
# - Finds and copies ALL referenced figures
# - Copies journal .odt files
# - Generates submission_checklist.md
```

#### Verification

The system creates `submit/submission_checklist.md` documenting:
- All files copied to submit folder
- Verification status of essential files
- Ready-for-submission confirmation

**⚠️ IMPORTANT**: 
- Files in `output/` are working files (NOT for submission)
- Files in `submit/` are final, ready-to-submit files
- NEVER submit directly from output/ folder
- See `prompts/prompts_submission_folder_mandatory` for complete details

## AI Peer Review System (ChatGPT 5 Pro & Gemini 2.5 Pro)

**⚠️ IMPORTANT**: AI reviewer prompts are for reviewing already-generated content, NOT for initial content generation.

### Two Review Options

#### Option 1: External Review (ChatGPT 5 Pro)
1. **Generate Paper with Claude Code**: First complete your paper sections or main.tex
2. **Copy Review Prompts**: Navigate to `prompts/prompts_Reviewer/`
3. **Submit to ChatGPT 5 Pro**: Use the prompts with your .tex files
4. **Receive Independent Review**: Get unbiased peer-review style feedback
5. **Save Reviews**: Ask Claude Code to save the reviews to `output/review_reports/`

#### Option 2: Internal Review (Claude Code)
1. **Generate Paper with Claude Code**: First complete your paper sections or main.tex
2. **Request Review**: Ask Claude Code "Please review my [section].tex using AI reviewer prompts"
3. **Receive Review**: Claude Code performs the review
4. **Review Saved**: Automatically saved to `output/review_reports/`
5. **Implement Feedback**: Decide on revisions based on review

### Available Review Types

1. **Section-Specific Reviews** - Evaluate individual sections:
   - `prompts_review_abstract` - Abstract evaluation
   - `prompts_review_introduction` - Introduction assessment
   - `prompts_review_methods` - Methods scrutiny
   - `prompts_review_resultsDisussions` - Results/Discussion analysis
   - `prompts_review_conclusions` - Conclusions review
   - `prompts_review_reference` - Bibliography quality check (uses Playwright MCP for verification)

2. **Full Manuscript Review** - Comprehensive evaluation:
   - `prompts_review_main` - Reviews complete main.tex
   - Produces executive summary, major/minor concerns
   - Technical and formatting audit
   - Recommendation (Accept/Minor Rev/Major Rev/Reject)
   - **NEW**: Generates formal referee report (referee_report_v1.tex/pdf)
   - Includes revision roadmap for Major Rev/Reject decisions

3. **Revision Support** - Address review feedback:
   - `prompts_making_corrections_single_section` - Section revisions
   - `prompts_making_corrections_entire_manuscript` - Full revision with response letter

### Review Storage
```
output/
└── review_reports/         # Created when reviews are performed
    ├── introduction_peer_review_v1.md  # Review of introduction_v1.tex
    ├── introduction_peer_review_v2.md  # Review of introduction_v2.tex
    ├── methods_peer_review_v1.md      # Review of methods_v1.tex
    ├── methods_peer_review_v3.md      # Review of methods_v3.tex
    ├── main_peer_review_v1.md         # Review of main_v1.tex
    └── revision_v1.md                  # Revision tracking
```

**CRITICAL**: Review files MUST include the version number of the .tex file being reviewed!

### Usage Guidelines

**PROHIBITED Uses**:
- ❌ Using review prompts when writing new .tex files
- ❌ Using review prompts during initial paper generation
- ❌ Using review criteria to guide content creation

**ALLOWED Uses**:
- ✅ Reviewing already-generated .tex files
- ✅ External review via ChatGPT 5 Pro
- ✅ Internal review when user explicitly requests
- ✅ Saving and organizing review feedback

**REMEMBER**: Review prompts evaluate existing content; they do not generate new content.

## LaTeX Package Management & Synchronization

The system automatically tracks, manages, and synchronizes LaTeX package dependencies to prevent compilation failures:

### Package Synchronization (NEW August 11, 2025)
**MANDATORY before final compilation**: Synchronize packages from section wrappers to main.tex:
```bash
# Extract packages from all wrappers and compare with main.tex
cd output
python ../utilityScripts/extract_wrapper_packages.py

# If package_patch.tex is generated, add missing items to main.tex
# Common missing packages: amsthm, subcaption, multirow, xcolor
```

### Package Documentation
Each section file includes package requirements at the top:
```latex
% Package requirements for this section:
% - amsmath (for equations, align, matrices)
% - algorithm, algorithmic (for algorithm blocks)
% - subcaption (for subfigures)
% - booktabs (for professional tables)
```

### Automatic Package Detection
Before final PDF compilation, run the package detection script:
```bash
cd output
~/.venv/ml_31123121/bin/python ../utilityScripts/detect_required_packages.py

# This will:
# - Scan all section .tex files for LaTeX commands
# - Map commands to required packages
# - Compare with main.tex preamble
# - Generate package_requirements_report.json
# - Create missing_packages.txt if packages are missing
```

### Common Package Requirements
- **Methods Section**: `amsmath`, `amssymb`, `algorithm`, `algorithmic`
- **Results Section**: `graphicx`, `subcaption`, `booktabs`, `multirow`
- **All Sections**: Remember `\graphicspath{{figures/}}` after `graphicx`

### Workflow Integration
1. Write section → Document packages at top of file
2. Compile section PDF → Verify packages work
3. Before final assembly → Run detection script
4. Update main.tex → Add any missing packages
5. Compile final PDF → No package errors!

See `prompts/prompts_package_requirements` for detailed instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Innovation Features

### Breakthrough Development Strategies
1. **Theory Synthesis**: Combine 2-3 existing theories in novel ways
2. **Algorithm Fusion**: Merge complementary algorithms to overcome limitations
3. **Paradigm Shift**: Challenge fundamental assumptions in the field
4. **Cross-Domain Transfer**: Apply techniques from other disciplines
5. **Mathematical Reformulation**: Reframe problems in different mathematical spaces

### Innovation Validation
- Each paper section integrates breakthrough findings
- Introduction emphasizes research gaps and proposed innovation
- Methods detail the novel approach with mathematical rigor
- Results quantify improvements over state-of-the-art
- Systematic comparison with existing methods
- Title generation reflects breakthrough approach and innovation

## Enhanced Verification System

### Pre-Flight Testing
Before any paper collection begins, the system runs mandatory verification tests:
- Known good paper extraction test
- Forbidden pattern rejection test
- Malformed PDF handling test
- Title extraction validation
- Batch validation test

### Multi-Stage Verification Pipeline
1. **Pre-Download Verification**: Check metadata before downloading
2. **Post-Download Extraction**: Use multiple methods (pdfplumber, PyMuPDF, Tika)
3. **Cross-Reference Validation**: Compare extracted content with expected metadata

### Forbidden Patterns and Circuit Breakers
- **Forbidden Author Patterns**: "Unknown Author", "Anonymous", "N/A", "[Author Name]"
- **Forbidden Title Patterns**: "Draft version", "arXiv:", "Untitled", "[Title]"
- **Circuit Breakers**:
  - 5 consecutive "Unknown Authors" → Stop and review
  - 10 consecutive verification failures → Stop web scraping
  - Same error pattern 3+ times → Switch extraction method
  - Total failure rate >25% → Abort and request intervention

### Batch Validation Rules
- >10% unknown authors → STOP entire process
- >20% similar titles → Likely extraction error → STOP
- >30% domain failures → Wrong search terms → STOP
- >15% empty abstracts → PDF extraction failure → STOP

### Verification Reports
- Real-time dashboard (`verification_dashboard.json`)
- Per-paper verification logs
- Human-readable summary reports
- Critical alerts for emergency situations

## Reference Quality Standards

### Strict Bibliography Requirements
- **NO "Unknown Authors"**: Papers with "Unknown Authors", "Anonymous", or similar are STRICTLY FORBIDDEN
- All citations must have identifiable authors
- Papers failing author verification are automatically excluded
- Special logging for rejected "Unknown Authors" papers
- Final verification checks for any anonymous entries
- **Bibliography sanity checks** detect suspicious patterns and author concentration
- **Kill Switch**: If ANY paper has "Unknown Authors" → DELETE immediately → Log removal → Search alternative

## Code Execution Handoff Process

The system uses an enhanced handoff process with mandatory validation and tracking:

### Immediate Handoff Triggers
- **Bash command timeout** (10 minutes maximum) → IMMEDIATE HANDOFF
- **ANY machine learning model training** → IMMEDIATE HANDOFF
- **Iterative algorithms >100 iterations** → IMMEDIATE HANDOFF
- **Scripts estimated >10 minutes runtime** → IMMEDIATE HANDOFF

See `prompts/prompts_code_execution_instructions` and `prompts/prompts_code_timeout_protocol` for concrete examples.

### Enhanced Handoff Workflow
1. **Development Phase**: System develops scripts with comprehensive test modes
2. **Validation Phase**: 
   - Run `validation_checklist.py` to verify all scripts
   - Test edge cases (empty data, NaN, extreme values)
   - Verify imports and syntax in correct environment
3. **Planning Phase**:
   - Generate `execution_plan.json` with dependency analysis
   - Create execution DAG to determine script order
   - Identify parallel execution opportunities
4. **Tracking Setup**:
   - Create `execution_status.json` for progress monitoring
   - Generate `execution_tracker.py` for status updates
   - Enable pause/resume capability
5. **Instructions Generation**:
   - Create `EXECUTION_INSTRUCTIONS.md` with step-by-step guide
   - Include runtime estimates and troubleshooting
   - Document all expected outputs
6. **User Execution**: User follows instructions and updates progress
7. **Result Integration**: Results/figures are generated in output folders

### Kill Switch for Version Creation
**CRITICAL**: System will NOT attempt to fix timeouts by:
- ❌ Creating v2/v3 with fewer epochs
- ❌ Reducing data size to avoid timeout
- ❌ Simplifying algorithms to run faster

Instead, system will IMMEDIATELY handoff to user for execution.

### Validation and Tracking Tools
The system creates these helper scripts for code execution:

1. **validation_checklist.py**: Comprehensive script validation
   - Syntax checking for all Python files
   - Import verification in correct environment
   - Test mode execution verification
   - Edge case handling checks
   - Progress indicator validation
   - Generates `validation_report.json`

2. **generate_execution_plan.py**: Dependency analysis and planning
   - Analyzes input/output dependencies between scripts
   - Creates execution DAG (Directed Acyclic Graph)
   - Identifies parallel execution opportunities
   - Estimates runtime for each script
   - Generates `execution_plan.json`

3. **execution_tracker.py**: Progress monitoring
   - Tracks script execution status (not_started/running/completed/failed)
   - Records start/end times and duration
   - Verifies output generation
   - Updates `execution_status.json`
   - Provides real-time progress dashboard

4. **EXECUTION_INSTRUCTIONS.md**: Step-by-step guide
   - Prerequisites checklist
   - Numbered execution steps in dependency order
   - Commands for each script with full paths
   - Expected inputs/outputs documentation
   - Troubleshooting guide for common issues
   - Progress tracking instructions

## Code Execution & Testing Standards

### Test Result Display Format
All validation and test results use consistent visual indicators:
- ✓ = PASSED - Test or validation succeeded
- ✗ = FAILED - Test or validation failed

Example output:
```
✓ Syntax check: PASSED
✓ Import verification: PASSED
✗ Runtime test: FAILED
✓ Output generation: PASSED
```

## Code Execution Summary Documentation

The system automatically generates comprehensive summaries after running or debugging scripts:

### Post-Execution Summaries
After successfully running ANY script, the system creates `{script_name}_v{N}_summary.md` containing:
- **Execution details**: Timestamp, runtime, environment, status
- **Input files used**: All data, models, configs accessed
- **Output files generated**: Organized by category
  - 📊 Figures (PNG, JPG, PDF) in `output/figures/`
  - 📈 Data files (CSV, JSON, NPY) in `output/data/`
  - 🤖 Model files (PT, PKL, H5) in `output/models/`
  - 📄 Log files (TXT, LOG) in `output/logs/`
- **Key results**: Performance metrics, accuracy, improvements
- **Next steps**: Integration guidance for paper sections

### Debug Summaries
When debugging scripts, the system creates `{script_name}_v{N}_debug_summary.md` containing:
- **Issues identified**: Detailed error descriptions and fixes
- **Performance improvements**: Memory optimization, speedup metrics
- **Validation results**: 10-minute test run confirmation
- **Files ready for execution**: Debugged scripts with execution commands
- **Major improvements**: Robustness, portability, efficiency gains

### Summary Benefits
1. **Traceability**: Know exactly what each script version produced
2. **Documentation**: Automatic documentation of all outputs
3. **Debugging Aid**: Compare outputs between versions
4. **Progress Tracking**: See what's been completed
5. **Report Generation**: Easy to include results in papers

### Example Usage
```bash
# After running analysis_v3.py
cat output/codes/analysis_v3_summary.md

# After debugging train_model_v13.py to v14
cat output/codes/train_model_v14_debug_summary.md
```

See `prompts/prompts_code_execution_summary` for templates and automation tools.

## Long-Running Script Monitoring System

The system provides comprehensive monitoring for scripts that exceed Claude Code's 10-minute execution limit, enabling asynchronous debugging through file-based communication.

### When Monitoring is Required
- **Training loops** with epochs > 5
- **Large iterations** (>1000)
- **ML model training** (any epochs)
- **Web scraping** (>20 URLs)
- **Data processing** (>1000 rows)
- **Any script** estimated >10 minutes runtime

### Monitoring Components

#### 1. Status Tracking (`logs/script_status.json`)
```json
{
  "status": "running|completed|error",
  "script_name": "train_model_v3.py",
  "current_stage": "epoch_5_of_100",
  "progress_percentage": 5,
  "estimated_remaining": "3 hours"
}
```

#### 2. Error Logging (`logs/error_log.json`)
Captures comprehensive error details including:
- Stack traces
- System resource usage
- Recovery suggestions
- Error context

#### 3. Progress Logging (`logs/progress_log.txt`)
Detailed timestamped progress updates for tracking execution flow.

#### 4. Checkpoint System
Save and resume from checkpoints for long-running operations.

### Usage Instructions

#### Running with Monitoring
```bash
# Option 1: Universal Runner (Recommended)
python utilityScripts/runner.py python train_model_v3.py --epochs 100

# Option 2: Bash Wrapper
bash utilityScripts/monitor_wrapper.sh train_model_v3.py --epochs 100

# Option 3: Integrate in Script
# Add MonitoredScript base class to your script
```

#### Checking Status
```bash
# Check status once
python utilityScripts/check_status.py

# Monitor continuously (every 20 minutes)
python utilityScripts/check_status.py --continuous

# Check specific log directory
python utilityScripts/check_status.py --log-dir ./output/logs
```

#### Debugging Failed Scripts
1. **Collect logs** when script fails:
   - `logs/script_status.json`
   - `logs/error_log.json`
   - `logs/execution.log` (last 200 lines)

2. **Return to Claude Code** with logs for debugging

3. **Resume from checkpoint** if available:
   ```bash
   python train_model_v3.py --resume logs/checkpoints/checkpoint_epoch_20.pth
   ```

### Monitoring Utilities

| Script | Purpose |
|--------|---------|
| `monitoring_base.py` | Base class for monitored scripts |
| `check_status.py` | Check script execution status |
| `runner.py` | Universal script runner with monitoring |
| `monitor_wrapper.sh` | Bash wrapper for monitoring |

### Integration with Handoff
When scripts require user execution:
1. Claude Code adds monitoring to scripts
2. Provides monitoring instructions and explains limitations
3. User runs script with monitoring (and optionally monitoring assistant)
4. User periodically checks status or waits for assistant alerts
5. If failure occurs, user returns logs to Claude Code
6. Claude Code debugs using logs and provides solutions

**Key Point**: Claude Code cannot check status autonomously - users must initiate all interactions.

See `prompts/prompts_monitoring_limitations` for capabilities and limitations.
See `prompts/prompts_long_running_monitoring` for implementation details.
See `prompts/prompts_asynchronous_debugging` for debugging procedures.

## Internal Process Checklists

The system follows strict internal checklists to ensure quality:

### Section Completion Checklist
1. Finish writing .tex file
2. Add review checklist (for review PDFs)
3. Extract section-specific citations
4. Create wrapper with section-specific .bib
5. **Verify all figure paths exist** (run verify_figures_v1.py)
6. Compile the PDF
7. **Rename wrapper PDF** (e.g., introduction_v1_wrapper.pdf → introduction_v1.pdf)
8. Validate figures with boxes
9. Present to user
10. Receive approval before continuing

### Final Assembly Checklist
1. Run version detection with content analysis
2. Create final_sections_summary.md
3. Review summary for correctness
4. Present summary to user for approval
5. Receive user confirmation
6. Merge bibliographies
7. Create clean versions without checklists

## System Integrity Verification (August 2025)

### Verification Status (Updated Aug 10, 2025)
- ✓ **CLAUDE.md Size**: 39,801 characters (well under 40k limit)
- ✓ **Prompt Consistency**: No conflicts found between prompt files
- ✓ **Reference Integrity**: All prompt references in CLAUDE.md are valid
- ✓ **Example Organization**: All concrete examples properly located in prompt files
- ✓ **Workflow Dependencies**: Sequential workflows properly documented
- ✓ **Timeout Consistency**: All files use 600 seconds (10 minutes) consistently
- ✓ **Version Naming**: Uniform enforcement across all prompts
- ✓ **Bibliography Validation**: Consistent "Unknown Authors" policies
- ✓ **Citation Styles**: Minor terminology variations identified and documented
- ✓ **User-Provided Materials**: Workflow consistently documented across all files
- ✓ **Output Directories**: Consistently use output/figures/ and output/data/ (no subdirectories)
- ✓ **Methods-Code Bidirectional**: Workflow consistently documented

### Key Design Principles
1. **CLAUDE.md as Index**: Serves as streamlined navigation to detailed prompts
2. **Prompt Modularity**: Each prompt file handles specific functionality
3. **No Redundancy**: Instructions appear in exactly one location
4. **Clear Dependencies**: Workflows reference prerequisite steps explicitly
5. **Version Consistency**: All related files use matching version numbers

### August 2, 2025 Optimization Summary
- **Character Reduction**: CLAUDE.md reduced from 39,991 to 36,250 chars (9.4% reduction)
- **Examples Relocated**: Concrete examples moved to appropriate prompts files:
  - Long-running indicators → `prompts_execution_preflight_check`
  - Prohibited content patterns → `prompts_paper_content_verification`
  - Bibliography validation patterns → `prompts_bibliography_validation`
  - Hallucination prevention → `prompts_code_before_text`
  - Version naming examples → `prompts_version_naming_rules`
- **Consistency Verified**: All prompts files checked for conflicts - none found
- **Instructions Streamlined**: CLAUDE.md now uses references instead of inline examples
- **NEW: Ultrathink Integration**: Added "Ultrathink with step-by-step reasoning" concept to 7 critical prompts files:
  - `prompts_methods` - For developing complex mathematical models and algorithms
  - `prompts_code_execution_instructions` - For complex code development and debugging
  - `prompts_research_breakthrough` - For deep gap analysis and innovation
  - `prompts_statistical_verification` - For rigorous statistical analysis
  - `prompts_resultsAndDiscussions` - For comprehensive results analysis
  - `prompts_code_debugging_protocol` - For systematic debugging
  - `prompts_methods_code_bidirectional` - For theory-implementation evolution

### August 14, 2025 Updates - Duplicate Reference Prevention

#### New Feature: Smart Bibliography Deduplication
- **Added Duplicate Reference Prevention Protocol** to CLAUDE.md with comprehensive detection methods
- **Created prompts_bibliography_deduplication**: Complete implementation with smart content-based deduplication
- **Enhanced prompts_section_compilation**: Added mandatory duplicate check before section compilation
- **Enhanced prompts_final_assembly**: Replaced simple merge with intelligent deduplication merge
- **Updated prompts_workflow_checklist**: Added duplicate check requirements to bibliography workflows

#### Key Features of Deduplication System:
- **Multiple Detection Methods**: DOI-based (most reliable), title similarity (>85% threshold), author-year matching
- **Smart Resolution**: Automatically chooses the most complete entry when duplicates found
- **Kill Switch**: Automatic halt if >5% duplicates detected
- **Deduplication Reports**: Generates detailed reports showing all duplicate resolutions
- **Citation Key Mapping**: Maintains consistency across version transitions

#### Files Created/Modified:
- **NEW**: `prompts/prompts_bibliography_deduplication` - Complete deduplication implementation
- **UPDATED**: `CLAUDE.md` - Added Duplicate Reference Prevention Protocol section
- **UPDATED**: `prompts/prompts_section_compilation` - Added duplicate check step
- **UPDATED**: `prompts/prompts_final_assembly` - Implemented smart merge
- **UPDATED**: `prompts/prompts_workflow_checklist` - Added duplicate check reminders

### August 14, 2025 Updates - Literal Interpretation Protocol

#### New Feature: Exact Information Usage System
- **Added Literal Interpretation Protocol** to CLAUDE.md with Three Laws of Information Handling
- **Created comprehensive prompt suite** for preventing unwanted additions and modifications
- **Enhanced precision** in bibliography creation, figure preservation, and file modifications

#### Key Features of Literal Interpretation System:
- **Verbatim Usage**: User-provided information used EXACTLY as given, no additions
- **No Assumptions**: Never add publishers, organizations, or missing fields not provided
- **Surgical Modifications**: Change ONLY what's explicitly requested in file updates
- **Clean Rebuild Discipline**: Perfect copy first, then minimal requested changes only
- **Figure Preservation**: Strict enforcement of figure count and order during revisions

#### Files Created/Modified:
- **NEW**: `prompts/prompts_exact_information_mandatory` - Enforce verbatim info usage
- **NEW**: `prompts/prompts_figure_preservation_strict` - Strict figure preservation rules
- **NEW**: `prompts/prompts_version_modification_protocol` - Surgical precision for edits
- **NEW**: `prompts/prompts_bibliography_entry_strict` - Bibliography with ONLY provided data
- **NEW**: `prompts/prompts_clean_rebuild_protocol` - Clean version rebuild procedures
- **UPDATED**: `CLAUDE.md` - Added Literal Interpretation Protocol section

### August 3-4, 2025 Updates
- **Large Figure Collection Handling**: Added comprehensive guidelines for handling 20-100+ figures with mandatory inclusion rules
- **Enhanced prompts_resultsAndDiscussions**: Added detailed organization strategies for large figure collections
- **No Editorial Decisions Rule**: System must include ALL user-provided figures, never select "representative" ones
- **Figure Verification Protocol**: NEW mandatory pre-compilation verification of all figure paths
- **Section PDF Naming Fix**: Clarified that wrapper PDFs must be renamed (e.g., section_v1_wrapper.pdf → section_v1.pdf)
- **New prompts_figure_verification**: Created comprehensive figure verification procedures with copy scripts
- **Enhanced prompts_section_compilation**: Added explicit PDF renaming instructions with warnings
- **User-Provided Materials**: Moved detailed workflow from USER_MANUAL.md to README.md for better visibility
- **Consistency Verification**: Comprehensive check of all prompts files confirmed no conflicts
- **CLAUDE.md Status**: Updated to 38,708 characters (well under 40k limit)
- **Key Confirmations**:
  - All timeout values consistently use 600 seconds (10 minutes)
  - Output directories consistently specified as output/figures/ and output/data/
  - Version naming rules uniformly enforced
  - Methods-Code bidirectional workflow consistently documented
  - User-provided materials workflow integrated across all relevant files
  - Large figure collection handling enforced in Results section

### January 12, 2025 Updates

#### Automatic Calculation Report Generation (NEW FEATURE)
- **What's New**: All computational scripts now automatically generate `report_calculation_v<N>_v1.md` files documenting workflow and results
- **Cross-referencing System**: Results sections must reference calculation reports when interpreting figures/tables
- **Files Updated**:
  - `prompts_methods` - Added mandatory report generation with implementation examples
  - `prompts_resultsAndDiscussions` - Added cross-referencing requirements  
  - `prompts_code_execution_summary` - Updated for dual documentation
  - `prompts_workflow_checklist` - Added report generation to checklists
  - `prompts_code_execution_instructions` - Added report verification step
  - `CLAUDE.md` - Added brief reference (kept under 40k chars)
- **Benefits**:
  - Improved traceability between code execution and paper content
  - Automated capture of key results for paper writing
  - Clear connection between methods and results sections

#### Clean File Page Control Commands Fix (BUG FIX)
- **Issue Fixed**: `\clearpage` commands were incorrectly appearing in clean tex files, violating LaTeX structure requirements
- **Root Cause**: Page control commands from main content weren't being removed during clean file generation
- **Files Updated**:
  - `CLAUDE.md` - Added explicit prohibition of page control commands in clean files
  - `prompts_clean_file_environment_rules` - Added comprehensive prohibited commands list and enhanced verification
  - `prompts_clean_section_files` - Added automatic removal of page control commands and examples
  - `prompts_section_compilation` - Clarified that `\clearpage` is ONLY for review checklists
  - `prompts_workflow_checklist` - Added verification step for page control commands
- **Prevention**:
  - Automatic detection and removal of `\clearpage`, `\newpage`, `\pagebreak` from main content
  - Enhanced validation script `verify_clean_files_v1.py` to check for prohibited commands
  - Clear documentation of what belongs in clean files vs main.tex

#### Conflict Resolution and Consistency Check
- **Maintenance**: Reviewed and resolved potential conflicts between CLAUDE.md and prompts files  
- **Consistency Verification Results**:
  - Code Execution Timeout: Confirmed consistent (10-minute limit for debugging only)
  - Report Generation: New feature integrated without conflicts
  - Bibliography Verification: Timing requirement "AFTER EVERY SECTION COMPILATION" consistent
  - Examples Location: All code examples verified in prompts files, not CLAUDE.md
  - Version Selection: Added clarification about defaulting to highest version number
- **Files Updated**:
  - `CLAUDE.md` - Added version selection clarification (default to highest version)
- **Key Principles Confirmed**:
  - CLAUDE.md instructions prevail when conflicts exist
  - Examples stay in prompts files to maintain <40k char limit
  - 10-minute execution for debugging/validation only
  - Final scripts run to completion without timeout

### August 11, 2025 Updates (Enhanced)

#### Package Synchronization System (CRITICAL FIX)
- **Problem Identified**: Package dependency loss during final assembly causing missing theorems, broken subfigures, and undefined citations
- **Root Cause**: Individual section wrappers contain essential LaTeX packages not transferred to main.tex
- **Solution Implemented**: Mandatory package synchronization workflow before final compilation

#### Files Created/Updated
- **NEW**: Created `prompts_package_synchronization` - Comprehensive package synchronization procedures
  - Extract packages from ALL wrapper files
  - Compare with main.tex packages
  - Generate package_patch.tex with missing items
  - Verification scripts and recovery procedures
- **UPDATED**: CLAUDE.md - Added "Package Synchronization for Final Assembly" section (optimized to 39,912 chars)
- **UPDATED**: `prompts_final_assembly` - Enhanced Step 1.5 with mandatory package synchronization
- **UPDATED**: `prompts_clean_section_files` - Added clarification that clean files do NOT handle packages
- **UPDATED**: `prompts_workflow_checklist` - Added package synchronization checklist before assembly

#### Key Implementation Details
- **Automatic Detection**: Script scans wrappers for \usepackage, \newtheorem, \newcommand, \graphicspath
- **Critical Packages**: amsthm (theorems), subcaption (subfigures), multirow/xcolor (tables), graphicspath (figures)
- **Stop Gates**: Compilation blocked if package_patch.tex exists with missing packages
- **Documentation**: Package_sync_report.md documents all synchronized packages

#### System Consistency Verification
- **Verified**: All prompts files are internally consistent with no conflicts
- **Verified**: Package management workflow properly separated into documentation, synchronization, and compilation phases
- **Verified**: Clean file generation and package synchronization are correctly independent processes
- **Optimized**: CLAUDE.md reduced from 39,990 to 39,912 characters by moving compilation examples to prompts files
- **Fixed**: Corrected date inconsistency (February → August) in Package Synchronization section

#### Submit Folder Workflow (NEW - Enhanced August 11, 2025)
- **Added**: Step 10 to final assembly workflow for copying files to submit folder
- **Purpose**: Organize all submission-ready files in one location
- **Key Features**:
  - Copies clean tex files (without review checklists) to existing submit/ folder
  - Copies main.tex, main.pdf, and ref_final.bib
  - Automatically finds and copies all referenced figures from multiple locations
  - **ENHANCED**: For journal papers, automatically copies cover_letter.odt and highlights.odt if they exist
  - Creates submission_checklist.md to track copied files
  - **IMPORTANT**: Uses existing submit/ folder - does NOT create new one
- **Updated Files (August 11)**:
  - `prompts_final_assembly` - Enhanced Step 10 with Step 6 for journal submission documents
  - `prompts_workflow_checklist` - Added journal documents to submit folder checklist
  - `CLAUDE.md` - Updated to reference journal docs in submit folder preparation (optimized to 39,965 chars)

#### Journal Submission Documents in Submit Folder (ENHANCED)
- **Enhancement**: Submit folder now automatically includes journal submission documents
- **Files Included**: 
  - `cover_letter.odt` - Cover letter to editors
  - `highlights.odt` - Research highlights
- **Automatic Detection**: System checks if these files exist and copies them if present
- **Benefits**:
  - Complete submission package in one location
  - No manual copying needed for journal documents
  - Submission checklist tracks all document types
- **Implementation**: Added Step 6 in prompts_final_assembly for journal document copying

#### Impact
This fix prevents:
- Theorems disappearing in final PDF (missing amsthm package)
- Subfigures breaking (missing subcaption package)  
- All citations showing [?] (bibliography issues)
- Figures not found (missing graphicspath setting)
- Content that rendered correctly in section PDFs failing in main.pdf

### August 10, 2025 Updates

#### Latest Version Enforcement for Conclusions and Abstract
- **Problem Addressed**: Conclusions and Abstract sections may reference outdated versions of other sections
- **Solution Applied**: Enforced mandatory use of LATEST versions when writing these sections

#### Files Updated
- **UPDATED**: `prompts_conclusions` - Added mandatory requirement to use LATEST methods_v*.tex and resultsAndDiscussions_v*.tex
- **UPDATED**: `prompts_summaryOrAbstract` - Added mandatory requirement to use LATEST versions of ALL section files  
- **UPDATED**: `prompts_workflow_checklist` - Added pre-writing checklists for Conclusions and Abstract sections
- **UPDATED**: CLAUDE.md - Added critical note about using latest versions (maintained under 40k limit)

#### Rename ChatGPTO3Pro to ChatGPT5Pro
- **RENAMED**: Folder `prompts_ChatGPTO3ProReview` → `prompts_ChatGPT5ProReview`
- **UPDATED**: All references to ChatGPTO3ProReview changed to ChatGPT5ProReview
- **UPDATED**: All references to "ChatGPT O3 Pro" changed to "ChatGPT 5 Pro"
- **CONSISTENCY**: Updated across CLAUDE.md, README.md, and all prompt files

#### Clarifications and Consistency Improvements
- **CLARIFIED**: Version selection protocol handles EXISTING files with legacy naming patterns (`_final`, `_revised`, etc.) but NEVER creates new files with these patterns
- **ENHANCED**: Figure path clarification - files stored in `output/figures/` but referenced as `figures/` in LaTeX due to `\graphicspath` setting
- **VERIFIED**: All prompts files are consistent - no conflicting guidelines found
- **VERIFIED**: CLAUDE.md remains at 39,801 characters (under 40k limit)

#### Key Implementation Details
- **Conclusions Section**: Must identify and use highest version numbers of methods and results sections
- **Abstract/Summary Section**: Must identify and use highest version numbers of ALL sections
- **Automatic Detection**: Writers must scan for v1, v2, v3... patterns and select highest number
- **Earlier Versions Ignored**: All previous versions considered outdated and must not be referenced

#### Impact
This update ensures:
- Conclusions accurately reflect final validated methods and results
- Abstract/Summary represents the final state of the entire paper
- No inconsistencies from referencing different versions
- Complete alignment between summary and actual content

### August 9, 2025 Updates

#### Abstract Environment Nesting Fix
- **Problem Identified**: Abstract was empty in main.pdf because abstract_v1_clean.tex contained `\begin{abstract}...\end{abstract}` wrapper commands, creating nested environment when main.tex also wrapped it
- **Solution Applied**: Created comprehensive environment nesting prevention system

#### Files Created/Updated
- **NEW**: Created `prompts_clean_file_environment_rules` - Comprehensive guide with:
  - Quick reference table for each file type
  - Verification scripts to check for nesting
  - Recovery procedures if issues occur
  - Clear examples of correct vs incorrect structure
- **UPDATED**: `prompts_clean_section_files` - Added Environment Nesting Prevention section with automatic abstract wrapper removal
- **UPDATED**: `prompts_final_assembly` - Added verification steps and enhanced clean file generation for abstract handling
- **UPDATED**: `prompts_workflow_checklist` - Added checklist items for abstract environment verification
- **UPDATED**: CLAUDE.md - Added Clean File Structure Requirements section (kept under 40k limit at 39,879 chars)

#### Key Implementation Details
- **Automatic Fix**: Clean file generation scripts now automatically remove `\begin{abstract}` and `\end{abstract}` from abstract files
- **Verification**: Pre-compilation checks verify no environment nesting will occur
- **Quick Reference**: Table shows exactly what each file type should contain:
  - `abstract_clean.tex`: Content only (NO environment wrappers)
  - `introduction_clean.tex`: `\section{Introduction}` + content (keep section command)
  - Other sections: Keep `\section{}` commands, remove review checklists

#### Impact
This fix prevents:
- Empty abstracts in final PDFs
- Missing section content
- LaTeX compilation warnings about nested environments
- Confusion about clean file structure requirements

### August 8, 2025 Updates
- **AI Report Content Update**: Removed incorrect statements about peer review/editorial decision-making from AI reports
- **UPDATED**: Enhanced `prompts_AIReport` with explicit instructions NOT to include disclaimers about AI non-involvement in review/editorial processes
- **CLARIFIED**: AI usage reports should focus on authors' use of AI tools, not journal/competition review processes
- **REASON**: Editorial and peer review processes are handled by journals/competitions separately from author AI usage disclosures
- **NEW AI Tools Added**: Updated AI report to include Claude Code Opus (claude-opus-4-1-20250805) and ChatGPT 5 Pro (chatgpt-5-pro-2025-08-07)
- **UPDATED**: Both competition and journal paper AI reports now reference the latest AI tool versions

### August 7, 2025 Updates
- **Direct File Editing Protocol**: Added mandatory protocol to prevent creation of intermediate files when editing LaTeX
- **NEW**: Created `prompts_direct_editing_protocol` with kill switches and comprehensive examples
- **CRITICAL FIX**: Resolved issue where intermediate files (compile_*.tex, temp_*.tex) were being created instead of direct editing
- **CLAUDE.md Status**: Optimized to 39,585 characters (under 40k limit) with streamlined code execution section

### August 6, 2025 Updates
- **Review Checklist Implementation Fix**: Enforced MANDATORY requirement for BOTH comment (%%) AND visible (verbatim) versions in all section PDFs
- **NEW**: Created `prompts_checklist_implementation` with explicit templates, verification script, and kill switch protocols
- **UPDATED**: Enhanced `prompts_section_compilation` with clear two-step checklist implementation and pre-compilation verification
- **UPDATED**: Modified `prompts_review_checkpoint` to emphasize BOTH versions are mandatory, never just comments
- **ADDED**: `check_checklist_implementation_v1.py` verification script to enforce dual-version requirement
- **CRITICAL FIX**: Resolved systemic issue where only comment versions were being added, missing the visible PDF version
- **CLAUDE.md Status**: Optimized to 39,999 characters (under 40k limit) while adding checklist implementation section

### August 5, 2025 Updates
- **Journal Paper Section Structure Corrected**: Clarified that Methods and Results & Discussion sections CAN have 2-4 subsections for major logical divisions
- **Enhanced CLAUDE.md**: Updated "Journal Paper Section Structure" with correct subsection guidelines
- **Updated prompts_journal_paper_narrative_enforcement**: Clarified that 2-4 subsections are acceptable in Methods and Results sections
- **Updated prompts_resultsAndDiscussions**: Distinguished organization strategies - competition papers (subsections allowed) vs journal papers (2-4 major subsections with narrative flow)
- **Enhanced prompts_workflow_checklist**: Updated "Journal Paper Narrative Compliance Check" with corrected subsection limits
- **Key Structure for Journal Papers**:
  - Introduction: ZERO subsections (strict narrative)
  - Methods: 2-4 subsections for major methodology components
  - Results & Discussion: 2-4 major subsections for themes
  - Conclusions: ZERO subsections (strict narrative)
  - Abstract: ZERO subsections (flowing paragraphs)
  - NO \subsubsection commands allowed anywhere
  - Each subsection must be substantial (>3 paragraphs)

## Development History

For detailed implementation history and changelog, see [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md).

## 📊 Calculation Report Generation (NEW August 2025)

### Overview
All computational scripts now MUST automatically generate calculation reports that document the workflow, parameters, and results. These reports are then cross-referenced when writing the Results & Discussion section to ensure accurate interpretation.

### Key Features
1. **Automatic Generation**: Scripts generate `report_calculation_v{N}_v1.md` upon completion
2. **Workflow Documentation**: Reports detail the complete computational process
3. **Cross-Referencing**: Results section MUST reference these reports when interpreting figures/tables
4. **Traceability**: Every numerical claim can be traced back to calculation reports

### Implementation
```python
# At the end of any computational script
def generate_calculation_report(script_name, version, results_dict):
    """Generate markdown report for calculation results"""
    # Document workflow, parameters, results
    report_path = f"output/codes/report_calculation_v{version}_v1.md"
    # ... implementation details in prompts/prompts_methods
```

### Usage in Papers
```latex
% When describing results, reference the calculation report
Figure~\ref{fig:results} shows the convergence behavior documented
in report_calculation_v3_v1.md (Section 2.2), where the algorithm
converged after 47 iterations with final error of 1.3e-4.
```

See `prompts/prompts_methods` for implementation details and `prompts/prompts_resultsAndDiscussions` for cross-referencing guidelines.

## Recent Updates (December 2024 - February 2025)

### NEW: Mandatory Code Audit Protocol (February 2025)
**Critical Fix**: System now enforces strict code auditing before ANY Python code creation to prevent disconnected/duplicated implementations:
- **The Three Laws**: 1) Reuse existing functionality, 2) Extend rather than recreate, 3) Maintain version continuity
- **Pre-Code Checklist**: MANDATORY audit of existing v* codebase before writing ANY code
- **Import-First Principle**: Every script MUST start with imports from existing v* codebase
- **Kill Switch**: Automatic halt if creating >50 lines without imports from v* codebase
- **Reviewer Response**: When addressing reviewer requests, MUST check existing code first
- Files modified:
  - `CLAUDE.md`: Added Code Audit Protocol section after Quick Start Guide
  - `prompts_workflow_checklist`: Added Pre-Code Creation Checklist section
  - `prompts_code_before_text`: Added Import-First Principle section
- New files created:
  - `prompts_code_audit_mandatory`: Comprehensive audit procedures and stop signs
  - `prompts_extend_not_recreate`: Extension patterns and decision matrix
  - `prompts_version_inheritance`: Version progression rules and compatibility patterns

### NEW: Bibliography & PDF Compilation Improvements (February 2025)
**Critical Fixes**: System now enforces proper bibliography handling and PDF naming to prevent compilation failures:
- **Bibliography-First Protocol**: MANDATORY creation of section bibliography BEFORE compilation
- **Caption Length Management**: Automatic detection and prevention of caption overflow (200 char limit)
- **PDF Naming Enforcement**: MUST create BOTH wrapper.pdf AND section.pdf files
- **Pre-Compilation Checks**: Automated validation of bibliography, captions, and figure paths
- Files modified:
  - `CLAUDE.md`: Added Bibliography-First Protocol, Caption Length Management, strengthened PDF naming
  - `prompts_section_compilation`: Added pre-compilation checks and automatic PDF renaming
- New files created:
  - `prompts_bibliography_compilation_protocol`: Step-by-step bibliography creation process
  - `prompts_pdf_naming_mandatory`: Enforces creation of both wrapper and user PDFs
  - `prompts_caption_length_validation`: Caption overflow prevention and fixes

### NEW: MCP vs Python Scripts Clarification (February 2025)
**Critical Clarification**: System documentation now properly distinguishes between Playwright MCP tools and Python virtual environments:
- **Playwright MCP**: Browser automation through MCP server infrastructure (NOT Python scripts)
- **Python Scripts**: Local file processing using virtual environments (`~/.venv/webScraping/`)
- **Key Insight**: MCP tools do NOT use Python - they run through Claude's infrastructure
- **Workflow Separation**: MCP for web operations, Python for local processing only
- Files modified:
  - `CLAUDE.md`: Added Web Scraping Methods Clarification section
  - `prompts_webScraping`: Added MCP vs Python distinction at the beginning
  - `prompts_workflow_competition/journal`: Clarified MCP is not Python in Phase 2
  - `prompts_technical_setup`: Reorganized to separate MCP tools from Python environments
  - `prompts_mcp_vs_python_clarification`: NEW comprehensive guide with decision trees and examples

### NEW: Pre-Writing Requirements Enforcement (February 2025)
**Critical Update**: System now enforces mandatory prerequisites before writing any section:
- **Introduction Blocking**: Auto-stops if <80 papers found; requires Playwright MCP for all searches
- **Playwright MCP Mandatory**: ALL paper searches must use mcp__playwright browser tools
- **Kill Switches**: Automatic process halts with no user prompting if prerequisites missing
- **Enhanced Verification**: Papers verified through Playwright browser snapshots and metadata extraction
- Files modified:
  - `CLAUDE.md`: Added Pre-Writing Requirements and BLOCKING REQUIREMENTS sections
  - `prompts_introduction`: Added mandatory prerequisite check script at beginning
  - `prompts_workflow_competition/journal`: Enhanced Phase 2 with Playwright MCP requirements
  - `prompts_workflow_checklist`: Added Literature Search Prerequisite Check section

### NEW: Figure Preservation and Reviewer Response Protocols (February 2025)
**Critical Fix**: System now prevents unintended figure modifications when implementing reviewer feedback:
- **Figure Preservation Protocol**: MANDATORY preservation of ALL figures unless explicitly requested
- **Reviewer Response Protocol**: MANDATORY comment classification before ANY implementation
- **Clean Version Usage**: Text-only changes MUST start from clean version with exact figure preservation
- **Figure Inventory Checks**: Automated verification of figure consistency between versions
- **Kill Switches**: Automatic halt if figures modified without explicit reviewer request
- Files modified:
  - `CLAUDE.md`: Added Figure Preservation Protocol and Reviewer Response Protocol sections
  - `prompts_section_compilation`: Added figure inventory verification for revisions
  - `prompts_review_checkpoint`: Added mandatory reviewer comment classification
- New files created:
  - `prompts_figure_preservation_protocol`: Complete figure management during revisions
  - `prompts_reviewer_response_protocol`: Systematic approach to implementing feedback
  - `prompts_webScraping`: Made Playwright MCP usage mandatory at the top

### REVERTED: Diagram Technical Specifications (January 2025)
- **Changes Reverted**: Removed strict font size and spacing requirements for diagrams
- **Rationale**: Technical specifications were overly prescriptive and conflicted with flexible visual design needs
- **Files Deleted**:
  - `prompts_diagram_technical_specs`: Removed detailed technical requirements
  - `prompts_diagram_validation_script`: Removed validation system
- **Files Reverted**:
  - `CLAUDE.md`: Removed diagram technical specifications section
  - `prompts_diagram_decision_guide`: Removed mandatory validation workflow
  - `prompts_workflow_checklist`: Removed diagram validation checklist
  - `prompts_methods`: Removed technical specs references
  - `prompts_draw_methods_architecture`: Reverted to original visual requirements
- **Current Approach**:
  - Focus on professional visual quality
  - Emphasis on icons, gradients, and visual metaphors
  - Flexibility in sizing based on content needs
  - Box overlap validation still available in `utilityScripts/`

### NEW: Flexible Diagram Requirements (January 2025)
- **3-4 diagram requirement** replaces rigid "exactly three" diagrams specification
- **Always Required**: Architecture and Workflow diagrams (2 diagrams minimum)

### NEW: Introduction Section Method Synchronization (January 2025)
- **MANDATORY**: When writing Introduction section in Adaptive workflow, MUST use LATEST methods_v*.tex file
- **Critical for Competition Papers**: "Statement of Purpose and Our Strategy on the Problem" subsection must reference models from highest methods version
- **Critical for Journal Papers**: Strategy discussion in narrative must use LATEST methods content
- **Files Updated**:
  - `CLAUDE.md`: Added requirement to use LATEST methods when writing Introduction
  - `prompts_introduction`: Added explicit guidance with version check scripts
  - `prompts_workflow_checklist`: Added checklist items for latest methods identification
  - `prompts_workflow_competition`: Updated Step 5 with LATEST methods requirement
  - `prompts_workflow_journal`: Updated Step 5 with LATEST methods requirement
- **Context-Dependent**: Mechanism diagram (for complex algorithms) and/or Hierarchy diagram (for hierarchical systems)
- **Decision Criteria**: System characteristics determine optimal diagram count (3 or 4)
- **Visual Style MANDATORY**: Illustrative elements including icons, gradients, visual metaphors (not just boxes)
- **New file created**: `prompts_diagram_decision_guide` with decision tree, examples, and implementation code
- **Files updated**: 
  - `CLAUDE.md`: Added new diagram requirements section
  - `prompts_methods`: Clarified diagram selection criteria
  - `prompts_draw_methods_architecture`: Updated for 3-4 diagrams flexibility
  - `prompts_workflow_checklist`: Added context-dependent diagram checks
  - `prompts_Reviewer/prompts_review_methods`: Aligned review criteria
- **Implementation Examples**: Python code for domain-specific icons (robots for AI, gears for processing, DNA for biology)

### UPDATED: Enhanced Bibliography and Clean Files Workflow (July 2025)
- **NEW**: Explicit versioned bibliography files for each section (e.g., `introduction_refs_v1.bib`, `introduction_refs_v2.bib`)
- **NEW**: Clean section files generation removes review checklists for final compilation
- **NEW**: Final bibliography `ref_final.bib` merges all section-specific bibliographies
- **NEW**: `prompts_clean_section_files` provides automated clean file generation
- **UPDATED**: `prompts_section_compilation` now emphasizes versioned .bib files
- **UPDATED**: `prompts_final_assembly` includes clean file generation and ref_final.bib workflow
- **UPDATED**: Main.tex uses clean files (*_clean.tex) and ref_final.bib for final compilation

### NEW: BibTeX File Download for Every Paper (January 28, 2025)
- **NEW**: Mandatory download of .bib file for every downloaded paper PDF
- **NEW**: Store .bib files in `output/papers/paper_bib/` directory
- **NEW**: Rename .bib files using citation keys (e.g., `PENWARDEN2023112464.bib`)
- **NEW**: Verify BibTeX content matches PDF metadata
- **NEW**: Use downloaded .bib files as primary source for citation completeness
- **UPDATED**: `prompts_webScraping` includes detailed BibTeX download procedures
- **UPDATED**: Bibliography validation prioritizes downloaded .bib files

#### BibTeX Download Process
1. For every downloaded paper PDF, immediately download its .bib file
2. Extract citation key from the .bib file content (e.g., `@article{PENWARDEN2023112464,`)
3. Rename the .bib file using the citation key (e.g., `PENWARDEN2023112464.bib`)
4. Store in `output/papers/paper_bib/` directory
5. Log the mapping in `bibtex_mapping.json`

#### Publisher-Specific Instructions
- **ArXiv**: Use `https://arxiv.org/bibtex/{arxiv_id}`
- **Elsevier/ScienceDirect**: Download file like `S0021999123005594.bib`, extract key, rename
- **IEEE Xplore**: Use "Download Citations" button, select BibTeX
- **ACM Digital Library**: Click "Export Citation", choose BibTeX
- **Springer**: Find "Download citation", select ".bib (BibTeX)"
- **Nature/Science**: Click "Download Citation", select BibTeX

#### Verification Integration
- Downloaded .bib files are now the primary source for citation completeness
- Bibliography verification workflow checks for downloaded files first
- If a .bib file exists for a citation key, it's used instead of web searching
- This significantly reduces false positives and improves accuracy

#### Benefits
1. **Improved Accuracy**: Publisher-provided BibTeX ensures correct metadata
2. **Reduced Web Searches**: Less reliance on Playwright MCP for basic citation info
3. **Better Verification**: Can verify citations match downloaded papers exactly
4. **Time Savings**: Faster bibliography completion using pre-downloaded data
5. **Quality Assurance**: Publisher metadata is more reliable than web scraping

#### Updated Files
- `prompts/prompts_webScraping` - Added comprehensive BibTeX download instructions
- `prompts/prompts_bibliography_validation` - Modified to first check for downloaded .bib files
- `prompts/prompts_bibliography_verification_workflow` - Added check for downloaded .bib files
- `prompts/prompts_citation_completeness_check` - Added `--bib-dir` parameter to scripts
- `prompts/prompts_workflow_checklist` - Added checklist items for verifying downloaded .bib files
- `CLAUDE.md` - Updated research paper requirements and project structure
- The system allows up to 5% of papers to lack .bib files (some older papers may not have them)

### How the New Workflow Works:
1. **Section Compilation**: Each section version has its own .tex and .bib file (e.g., `methods_v2.tex` with `methods_refs_v2.bib`)
2. **Clean File Generation**: Before final assembly, create clean versions without review checklists (e.g., `methods_clean.tex` from `methods_v2.tex`)
3. **Bibliography Merging**: All section .bib files merge into `ref_final.bib`
4. **Final Assembly**: main.tex uses clean files and ref_final.bib

### UPDATED: AI Peer Review System (ChatGPT 5 Pro & Gemini 2.5 Pro) (July 2025)
- Added peer review prompts for quality assurance of generated content
- Section-specific review prompts for targeted feedback
- Full manuscript review following journal peer-review standards (including main.tex)
- Revision support with tracked changes and response letter generation
- Located in `prompts/prompts_Reviewer/`
- **UPDATED (Jul 22)**: Clarified access rules - prompts can be used by both ChatGPT 5 Pro and Claude Code
- **UPDATED (Jul 22)**: Claude Code can access these prompts ONLY when user explicitly requests peer review
- **UPDATED (Jul 22)**: Cannot be used during initial content generation, only for reviewing existing content
- **UPDATED (Jul 22)**: Updated all prompts and documentation to reflect dual-use capability while maintaining separation between generation and review phases
- **UPDATED (Jul 23)**: Added citation balance checking to all review prompts - ensures alternation between author-prominent and information-prominent citation styles
- **UPDATED (Jul 24)**: Methods review now focuses on theorem correctness, gap fulfillment, and visual representations rather than generic reproducibility
- **UPDATED (Jul 24)**: Added journal paper writing style guidelines - scientific narrative flow instead of excessive bullet points
- **UPDATED (Jul 24)**: Enhanced Methods review constraints - no computational plots allowed, but theorem/algorithm improvements always permitted
- **UPDATED (Jul 24)**: Added review validation system in `prompts_format_verification` to automatically detect and prevent forbidden review requests
- **UPDATED (Jul 24)**: Added forbidden review patterns to `prompts_review_checkpoint` with section-specific constraints and acceptable alternatives
- **UPDATED (Jul 24)**: Added explicit "et al." usage rule in `prompts_introduction` and `prompts_citation_methods` - use "et al." for 3+ authors, both names for 2 authors
- **UPDATED (Jul 25)**: Enhanced file versioning enforcement - ALL files except CLAUDE.md and README.md must use sequential versioning (v1, v2, v3...)
- **UPDATED (Jul 25)**: Added strict peer review naming convention - review files MUST include tex version: `introduction_peer_review_v2.md` for `introduction_v2.tex`
- **UPDATED (Jul 25)**: Enhanced GitHub repository reference rules - STRICTLY FORBIDDEN to include license information, commit hashes, or version tags in paper content
- **UPDATED (Jul 25)**: Strengthened file versioning enforcement - ALL files must use sequential versioning with mandatory pre-check workflow
- **UPDATED (Jul 25)**: Enhanced methods section mathematical rigor - NO HALLUCINATIONS allowed, every formula must have proper derivation or citation
- **UPDATED (Jul 25)**: Added comprehensive referee report generation - creates referee_report.tex and PDF with Accept/Minor Rev/Major Rev/Reject recommendations and detailed feedback
- **UPDATED (Jul 25)**: Added citation/reference validation to prevent question marks [?] in PDFs - checks for undefined citations, missing references, and malformed labels
- **UPDATED (Jul 25)**: Enhanced Results section revision handling - mandatory code generation/modification for new graphs and tables, NO HALLUCINATIONS allowed
- **UPDATED (Jul 25)**: Added specific reminders in review prompts about code requirements when requesting new visualizations or data tables
- **UPDATED (Jul 25)**: Added comprehensive anti-hallucination protocols in CLAUDE.md - Statistical Claims Verification, Peer Review Response Protocol, and Hallucination Prevention Protocol
- **UPDATED (Jul 25)**: Created `prompts_statistical_verification` with detailed workflows for generating quantitative content from actual computation
- **UPDATED (Jul 25)**: Created `prompts_code_before_text` enforcing Code-First principle for all peer review implementations
- **UPDATED (Jul 25)**: Enhanced review prompts to cover ALL quantitative content (not just graphs/tables) - includes statistics, metrics, analyses
- **UPDATED (Jul 25)**: Added comprehensive verification checklists to workflow_checklist for statistical claims, peer review responses, and hallucination prevention
- **UPDATED (Jul 25)**: Enhanced bibliography validation - prevents incomplete citations missing journal names, volume/issue/pages
- **UPDATED (Jul 25)**: Added `prompts_review_bibliography_validation` for deep reference verification using Playwright MCP
- **UPDATED (Jul 25)**: Created `prompts_bibliography_validation` with comprehensive validation protocol and automated scripts
- **UPDATED (Jul 25)**: Enhanced citation completeness checks in web scraping, section compilation, and final verification stages
- **UPDATED (Jul 25)**: Added mandatory bibliography verification workflow after each section compilation
- **UPDATED (Jul 25)**: Created `prompts_bibliography_verification_workflow` for thorough reference checking with Playwright MCP
- **UPDATED (Jul 25)**: Created `prompts_bibliography_replacement_workflow` for replacing fake references with genuine ones
- **UPDATED (Jul 25)**: Enhanced workflow checklist to include bibliography verification as mandatory step
- **NEW (Jan 25)**: Created `prompts_citation_completeness_check` with Playwright MCP integration to automatically find and complete missing citation metadata
- **NEW (Jul 28)**: Added mandatory BibTeX file download for every paper with citation key extraction and renaming
- **UPDATED (Jul 28)**: Enhanced `prompts_webScraping` with publisher-specific BibTeX download instructions
- **UPDATED (Jul 28)**: Bibliography validation now prioritizes downloaded .bib files from paper_bib directory
- **NEW (Jul 28)**: Added mandatory ASCII-only requirement for abstracts/summaries to prevent arXiv rejection
- **UPDATED (Jul 28)**: Enhanced `prompts_summaryOrAbstract` with comprehensive list of forbidden non-ASCII characters and ASCII alternatives
- **NEW (Jul 30)**: Added user-provided materials handling - automatically copies files from root folders to output subfolders
- **NEW (Jul 30)**: Created `prompts_user_provided_materials` with comprehensive guidelines for handling pre-existing files
- **UPDATED (Jul 30)**: Clarified code execution language - results/figures are in output folders, not "user-provided"
- **UPDATED (Jul 30)**: Enhanced workflow start checklists to include user materials check as first step
- **UPDATED (Jul 30)**: Strengthened narrative flow requirements for journal papers - NO bullet points in Introduction, Abstract, or Conclusions
- **UPDATED (Jul 30)**: Added explicit guidance to transform all lists into flowing paragraphs for journal papers (max 2 bullet points per page)
- **UPDATED (Jul 30)**: Enhanced `prompts_writing_style_journal` with comprehensive narrative flow guidance and examples
- **NEW (Jul 30)**: Added timezone handling requirement - all timestamps must use Taiwan Standard Time (Asia/Taipei, UTC+8)
- **NEW (Jul 30)**: Created `prompts_timezone_handling` with Python and bash implementations for consistent timestamp formatting
- **UPDATED (Jul 30)**: Updated all timestamp generation in prompts files to use Taiwan timezone
- **UPDATED (Jul 30)**: Modified introduction section length to MAXIMUM 3 A4-size pages (2-3 pages for competitions, 2.5-3 pages for journals)
- **UPDATED (Jul 30)**: Enhanced user-provided materials handling - skip code execution unless debugging requested, preserve version control
- **UPDATED (Jul 30)**: Clarified workflow adjustments when user provides data/codes/figures - system adapts automatically
- **UPDATED (Jul 30)**: Changed code execution timeout threshold from 30 seconds to 600 seconds (10 minutes, the maximum timeout in Claude Code)
- **NEW (Jul 30)**: Created `prompts_execution_preflight_check` with systematic timeout detection patterns and concrete examples
- **UPDATED (Jul 30)**: Optimized CLAUDE.md to stay under 40k character limit by moving examples to prompts files while maintaining all essential instructions
- **NEW (Jul 31)**: Created `prompts_file_creation_checklist` with mandatory pre-file creation validation to prevent versioning violations
- **UPDATED (Jul 31)**: Enhanced file versioning system with stronger enforcement mechanisms and common violation examples
- **UPDATED (Jul 31)**: Added Pre-File Creation Checkpoint to CLAUDE.md and workflow checklist for immediate validation
- **UPDATED (Aug 1)**: Enhanced Methods-Code Consistency - MUST read latest methods_v*.tex before ANY code modification
- **UPDATED (Aug 1)**: Added Methods-Code Consistency section to CLAUDE.md with mandatory verification workflow
- **UPDATED (Aug 1)**: Enhanced `prompts_methods` with detailed theorem-to-code mapping requirements and version tracking
- **UPDATED (Aug 1)**: Enhanced `prompts_code_execution_instructions` with comprehensive methods-code consistency verification section
- **UPDATED (Aug 1)**: Updated workflow checklist with methods-code alignment checkpoints
- **NEW (Aug 2)**: Enforced output file version matching - outputs MUST use same version as generating script (e.g., band_structure_v13.png from script v13)
- **UPDATED (Aug 2)**: Enhanced `prompts_version_naming_rules` with explicit script-output version matching requirements
- **UPDATED (Aug 2)**: Updated `prompts_code_execution_instructions` with output versioning examples and requirements
- **UPDATED (Aug 2)**: Enhanced `prompts_file_creation_checklist` to emphasize output version matching
- **UPDATED (Aug 2)**: Updated `prompts_draw_methods_architecture` to use versioned output filenames
- **NEW (Aug 2)**: Added mandatory code debugging protocol - run corrected scripts for 10 minutes before user handoff
- **NEW (Aug 2)**: Created `prompts_code_debugging_protocol` with comprehensive debugging procedures and TUI display requirements
- **UPDATED (Aug 2)**: Modified CLAUDE.md to add Code Debugging section with reference to debugging protocol
- **UPDATED (Aug 2)**: Enhanced `prompts_code_execution_instructions` with debugging requirements section
- **UPDATED (Aug 2)**: Added debugging checklist to `prompts_workflow_checklist` for mandatory debugging verification
- **UPDATED (Aug 2)**: Added debugging documentation examples to `prompts_file_creation_checklist`
- **NEW (Aug 2)**: Added runtime output file naming clarification - semantic naming allowed in timestamped directories
- **NEW (Aug 2)**: Created `prompts_output_file_naming` with decision tree for output file naming
- **UPDATED (Aug 2)**: Enhanced `prompts_version_naming_rules` to distinguish source files from runtime outputs
- **UPDATED (Aug 2)**: Added output file naming section to `prompts_code_execution_instructions`
- **NEW (Aug 2)**: Created `prompts_ml_file_conventions` for ML-specific naming patterns
- **UPDATED (Aug 2)**: Clarified that figures go to output/figures/, data to output/data/ (not subdirectories)
- **UPDATED (Aug 2)**: Modified CLAUDE.md to reference semantic naming flexibility in timestamped directories
- **NEW (Aug 2)**: Created `prompts_file_copy_version_rules` for preserving versions when copying files to shared directories
- **UPDATED (Aug 2)**: Added file copy version rules section to CLAUDE.md after version naming rules
- **UPDATED (Aug 2)**: Enhanced `prompts_workflow_checklist` with file copy verification checklist
- **NEW (Aug 3)**: Created `prompts_pre_handoff_validation` with mandatory pre-handoff validation protocol
- **NEW (Aug 3)**: Created `prompts_debugging_workflow` with exact 10-minute debugging requirements
- **NEW (Aug 3)**: Created `prompts_import_templates` with common import patterns and verification
- **UPDATED (Aug 3)**: Enhanced `prompts_code_execution_instructions` with pre-execution syntax checks and import verification
- **UPDATED (Aug 3)**: Added references to new debugging and validation prompts in CLAUDE.md
- **UPDATED (Aug 3)**: Optimized CLAUDE.md to stay under 40k character limit by consolidating content
- **NEW (Aug 3)**: Created `prompts_code_execution_summary` for documenting all outputs after script execution
- **NEW (Aug 3)**: Added mandatory debug summaries tracking fixes and improvements with sequential naming
- **UPDATED (Aug 3)**: Enhanced workflow checklist with post-execution summary requirements
- **UPDATED (Aug 3)**: Modified code debugging protocol to include mandatory summary documentation
- **UPDATED (Aug 2)**: Optimized CLAUDE.md from 39,971 to 36,250 characters for better performance
- **UPDATED (Aug 2)**: Enhanced prompt files with concrete examples moved from CLAUDE.md
- **UPDATED (Aug 2)**: Verified consistency across all prompts files - no conflicts found
- **UPDATED (Aug 2)**: Added comprehensive patterns to prompts_bibliography_validation, prompts_paper_content_verification, and prompts_code_before_text
- **NEW (Aug 2)**: Integrated "Ultrathink with step-by-step reasoning" concept into 7 critical prompts files for complex task handling

### NEW: File Copy Version Rules (August 2, 2025)
- **CRITICAL**: Files copied from timestamped directories to shared locations MUST have version numbers added
- **Problem Solved**: Prevents untraceable outputs when copying files like `band_structure_epoch_100.png`
- **Implementation**: Process files individually during copy to add version from script/directory name
- **Example**: `cp training_v24_*/result.png output/figures/` → `result_v24.png` (version preserved)
- **NEW**: Created `prompts_file_copy_version_rules` with bash/Python patterns and comprehensive examples
- **UPDATED**: CLAUDE.md includes mandatory file copy rules after version naming section
- **UPDATED**: Workflow checklist includes file copy verification steps
- This ensures complete traceability for all files in shared directories

### NEW: Methods-Code Bidirectional Workflow (August 2, 2025)
- **REVOLUTIONARY**: Methods and code now evolve together through iterative improvement
- **Initial Phase**: Code implementations follow methods.tex exactly as before
- **Discovery Phase**: When code experiments yield >10% improvement, create NEW methods_v{n+1}.tex
- **Theory Update**: Document theoretical basis for improvements in updated methods version
- **Consistency Maintained**: Both code and theory remain synchronized throughout evolution
- **NEW**: Created `prompts_methods_code_bidirectional` with comprehensive workflow details
- **UPDATED**: CLAUDE.md Methods-Code section now reflects bidirectional workflow
- **UPDATED**: `prompts_methods` includes bidirectional evolution examples
- **UPDATED**: `prompts_code_execution_instructions` shows discovery documentation
- **UPDATED**: `prompts_workflow_checklist` adds bidirectional tracking section
- This change enables breakthrough discoveries through empirical experimentation

### File Versioning System Enhancement (July 2025)
- **MANDATORY sequential version numbering** for ALL files (v1, v2, v3...)
- Added comprehensive `prompts_version_naming_rules` with concrete examples
- Updated all prompts to enforce version naming throughout workflow
- Prevents accidental file overwrites and enables full change tracking
- Forbidden patterns: `_final`, `_updated`, `_revised`, `_new`, `_modified`

### CLAUDE.md Optimization (July 2025)
- File size reduced from 41.2k to 39.4k characters (keeping under 40k threshold)
- Concrete examples and implementations moved to specialized prompt files for better organization
- Maintains effectiveness as streamlined index while significantly improving Claude Code performance
- **August 2, 2025 Update**: Further optimized from 39,971 to 36,250 chars by moving examples to prompts files
- References to prompt files provide quick access to detailed implementations
- All key instructions retained while examples are properly documented in corresponding prompts files

### Enhanced Prompt Files (July 2025)
The following prompt files now contain comprehensive examples previously in CLAUDE.md:
- `prompts_code_execution_instructions`: Timeout triggers, runtime formulas, validation scripts, tracking tools
- `prompts_webScraping`: Multi-stage verification pipeline, report generation, undownloadable papers protocol
- `prompts_paper_content_verification`: Storage structures, similarity calculations, recovery procedures
- `prompts_section_compilation`: Complete compilation commands, bibliography extraction, error recovery
- `prompts_final_assembly`: Section summary review, bibliography merging, post-compilation file listing script
- `prompts_package_requirements`: Full `detect_required_packages.py` implementation and package mapping
- `prompts_figure_box_validation`: Complete `box_overlap_checker.py` and `smart_box_layout.py` implementations
- `prompts_figure_verification`: Comprehensive figure path verification with `verify_figures_v1.py` and `copy_figures_with_version_v1.py` (NEW Feb 9)
- `prompts_version_naming_rules`: Comprehensive version naming examples and enforcement rules (UPDATED Aug 2)
- `prompts_resultsAndDiscussions`: Enhanced with mandatory figure inclusion rules for large collections (UPDATED Feb 8)
- `prompts_file_creation_checklist`: Mandatory pre-file creation validation checklist with kill switches (NEW Jan 31)
- `prompts_direct_editing_protocol`: Direct file editing enforcement with kill switches and decision trees (NEW Feb 12)
- `prompts_methods_code_bidirectional`: Comprehensive bidirectional workflow with decision trees and examples (NEW Aug 2)
- `prompts_citation_methods`: Detailed guide for alternating between author-prominent and information-prominent citation styles (NEW Jan 23)
- `prompts_writing_style_journal`: Scientific narrative flow guidance for journal papers - avoiding excessive bullet points (NEW Jan 24)
- `prompts_github_usage`: Enhanced guidelines for clean GitHub repository references without license/commit metadata, including real-world examples and common mistakes (UPDATED Jan 25)
- `prompts_citation_reference_validation`: Comprehensive validation to prevent [?] in citations and references (NEW Jan 25)
- `prompts_bibliography_validation`: Complete citation validation protocol with automated scripts and kill switches (NEW Jan 25)
- `prompts_citation_completeness_check`: Playwright MCP workflow to find and complete missing citation metadata (NEW Jan 25)
- `prompts_statistical_verification`: Complete workflows for generating quantitative content from computation, never fabrication (NEW Jan 25)
- `prompts_code_before_text`: Code-First protocol for peer review implementations - enforces data generation before text updates (NEW Jan 25)
- `prompts_user_provided_materials`: Comprehensive guidelines for handling user-provided files in root folders with copy procedures (NEW Jan 30)
- `prompts_output_file_naming`: Decision tree and guidelines for runtime output file naming with semantic flexibility (NEW Aug 2)
- `prompts_ml_file_conventions`: ML-specific file naming conventions for checkpoints, models, and artifacts (NEW Aug 2)
- `prompts_file_copy_version_rules`: Comprehensive guide for preserving version numbers when copying files to shared directories (NEW Aug 2)

This reorganization ensures optimal performance while preserving all functionality and enhancing maintainability.

## Recent Updates

See the "Recent Updates (December 2024 - August 2025)" section above for latest changes including the reversion of diagram technical specifications.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{lee2025scmspinn,
  title={Symmetry-Constrained Multi-Scale Physics-Informed Neural Networks for Graphene Electronic Band Structure Prediction},
  author={Lee, Wei Shan and Kwok, I Hang and Leong, Kam Ian and Chau, Chi Kiu Althina and Sio, Kei Chon},
  journal={arXiv preprint arXiv:2508.10718},
  year={2025},
  url={http://arxiv.org/abs/2508.10718}
}
```

## Acknowledgments

- Developed for use with Claude Code (claude.ai/code)
- Templates adapted from official competition and journal formats
- Enhanced with research breakthrough analysis for innovative paper generation
- Optimized for performance with streamlined CLAUDE.md and detailed prompt files
- Mandatory workflow checklists ensure consistent quality across all papers
