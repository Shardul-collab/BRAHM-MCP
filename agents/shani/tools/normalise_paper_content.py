import sqlite3, sys
DB_PATH = "/mnt/d/brahm/agents/shani/database/research_workflow.db"
DEFAULT_WORKFLOW_ID = 2
NOISE_EXACT = {"references","reference","referen_es","refernces","referencias","kaynaklar_references","references_cc","references_notes","references_and_notes","references_for_main_text","references_for_method_section","supplementary_references","viii_references","1_references","3_references","4_references","7_references","bibliography","acknowledgements","acknowledgement","acknowledgments","acknowledgment","v_acknowledgements","vi_acknowledgements","1_acknowledgements","agradecimientos","kno_wledgemen_ts","a_kno_wledgemen_ts","author_contributions","author_contribution","author_contribution_statement","author_declarations","author_information","authors_contribution","authors_contributions","author_details_1","contributions","yazarlarin_katkilari_authors_contributions","contribucin_de_los_autores","data_availability","data_availability_statement","data_availability_statements","code_availability","date_and_code_availability","availability_of_data_and_materials","competing_interests","competing_financial_interests","conflict_of_interest","conflicts_of_interest","declaration_of_competing_interest","declaration_of_interests","ikar_atimasi_conflict_of_interest","conflicto_de_intereses","funding","additional_information","supplementary_information","supplementary_material","supplementary_materials","supplementary_files","supporting_information","supporting_material","supporting","copyright","orcid","orcid_ids","open_access","reporting_summary","publishers_note","check_for_updates","just_accepted","just_accepted_j","correspondence","corresponding_author","corresponding_authors","correspondencia","edited_by","reviewed_by","specialty_section","reprints_and_permissions_information_is_available_at_http_wwwnaturecom_reprints","graphical_abstract","grafik_zet_graphical_abstract","highlights","nemli_noktalar_highlights","figure_captions","figures","tables","editors_summary","editorial_summary","keywords","abbreviations","declarations","citation","article","article_in_press","article_info","associated_content","preamble","use_of_ai_statement","declaration_of_generative_ai_and_aiassisted_technologies_in_the_writing_process","note_added_in_proof","ethics_statement_not_applicable_cep","declaration_of_ethical_standards","informed_consent","1_1_2_3_1_1_1","o_o_o_o","table_of_contents","table_of_content","list_of_figures","list_of_publications","committee","dissertation","co_authored_journal_publications","conference_presentations","journal_publications"}
NOISE_PREFIXES = ("received_","accepted_","supplementary_note_","supplementary_text_","supporting_information_","acknowledgements_","acknowledgments_","data_availability_","declaration_of_competing_interest_","author_contributions_","figure_","eq_")
NOISE_SUFFIXES = ("_references","_acknowledgements","_acknowledgments","_acknowledgement")
RENAME_EXACT = {"abstra_t":"abstract","abstrct":"abstract","i_abstract":"abstract","i_introduction":"introduction","ii_introduction":"introduction","1_introduction":"introduction","in_tro_du_tion":"introduction","introduction_j":"introduction","giri_introduction":"introduction","introduction_history_and_events":"introduction","ii_methods":"methods","2_methods":"methods","experimental":"methods","experimental_section":"methods","experimental_method":"methods","experimental_methods":"methods","experimental_procedure":"methods","experimental_details":"methods","experimental_setup":"methods","materials_and_methods":"methods","materials_and_device_fabrication":"methods","ii_results":"results","iii_results":"results","i_results":"results","result_and_discussion":"results_and_discussion","results_and_discussions":"results_and_discussion","ii_results_and_discussion":"results_and_discussion","iii_results_and_discussion":"results_and_discussion","2_results_and_discussion":"results_and_discussion","ii_discussion":"discussion","iii_discussion":"discussion","iv_discussion":"discussion","v_discussion":"discussion","14_discussion":"discussion","25_discussion":"discussion","4_discussion":"discussion","iii_conclusion":"conclusion","iv_conclusion":"conclusion","v_conclusion":"conclusion","vi_conclusions":"conclusion","vii_conclusions":"conclusion","iv_conclusions":"conclusion","1_conclusions":"conclusion","5_conclusion":"conclusion","con_luding_remarks":"conclusion","final_remarks":"conclusion","sonu_conclusion":"conclusion","conclusiones_y_perspectivas":"conclusion","conclusion_and_future_outlook":"conclusion","conclusion_and_perspective":"conclusion","conclusions_and_outlook":"conclusion","conclusions_and_perspectives":"conclusion","short_summary":"conclusion","summary_and_outlook":"conclusion","conclusion_this_work_expl":"conclusion","conclusion_to_conclude_a":"conclusion","synthesis":"synthesis","crystal_growth":"synthesis","cvd_growth":"synthesis","in2se3_synthesis":"synthesis","sample_synthesis":"synthesis","thin_film_growth":"synthesis","material_preparation":"synthesis","material_preparations":"synthesis","sample_preparation":"synthesis","sample_fabrication":"synthesis","growth_of_in2se3":"synthesis","in2se3_synthesis_using_mbe":"synthesis","1_material_preparation":"synthesis","1_preparation_and_characterization_of_in2se3_thin_film":"synthesis","1_thin_film_preparation_and_characterization_of_in2se3":"synthesis","characterizations":"characterization","structural_characterization":"characterization","material_characterization":"characterization","materials_characterization":"characterization","raman_characterization":"characterization","xrd_analysis":"characterization","optical_characterization":"characterization","electrical_characterization":"characterization","structural_and_morphological_characterization":"characterization","2_characterizations":"characterization","3_characterization":"characterization","1_structural_of_in2se3_properties":"characterization","12_xrd_raman_and_afm":"characterization","raman_spectroscopy":"characterization","raman_spectroscopy_analysis":"characterization","afm_measurements":"characterization","sem_measurements":"characterization","pfm_measurements":"characterization","optical_properties":"optical_properties","optical_analysis":"optical_properties","2_optical_calculation_of_in2se3_thin_film":"optical_properties","21_transmittance_and_reflectance_spectra":"optical_properties","22_refractive_index_and_dispersion_analysis":"optical_properties","23_dielectric_characterization":"optical_properties","24_the_nonlinear_optical_characteristics_of_the_in2se3_film":"optical_properties","device_fabrications":"device_fabrication","device_fabrication_and_measurements":"device_fabrication","device_fabrication_and_characterization":"device_fabrication","device_fabrication_and_electrical_measuremen":"device_fabrication","fabrication_and_experimental_details":"device_fabrication","devices_fabrication":"device_fabrication","dft_calculation":"computational_methods","first_principles_calculations":"computational_methods","density_functional_theory_calculations":"computational_methods","density_functional_theory_dft_calculations":"computational_methods","theoretical_calculations":"computational_methods","computational_details":"computational_methods","computational_methodology":"computational_methods","calculation_details":"computational_methods","electrical_measurement":"electrical_properties","electrical_measurements":"electrical_properties","transport_properties":"electrical_properties","electrical_transport_measurements":"electrical_properties"}
RENAME_PREFIX = [("chapter_1_introduction","introduction"),("chapter_2_literature","introduction"),("chapter_3_experimental","methods"),("chapter_4_","results"),("chapter_5_","results"),("chapter_6_","discussion"),("chapter_7_conclusion","conclusion"),("growth_of_","synthesis"),("synthesis_of_","synthesis"),("fabrication_of_","device_fabrication"),("device_fabrication_and","device_fabrication")]
RENAME_CONTAINS = [("results_and_discussions","results_and_discussion"),("results_and_discussion","results_and_discussion"),("synthesis_and","synthesis"),("device_fabrication","device_fabrication"),("computational_method","computational_methods"),("dft_calculation","computational_methods"),("first_principles","computational_methods")]
KEEP_PATTERNS = {"results","discussion","conclusion","introduction","methods","synthesis","characterization","computational","optical","electrical","device","abstract","summary","theory","experimental","fabrication","measurement","analysis"}
def is_noise(name):
    if name in NOISE_EXACT: return True
    for p in NOISE_PREFIXES:
        if name.startswith(p): return True
    for s in NOISE_SUFFIXES:
        if name.endswith(s): return True
    return False
def is_paper_title(name):
    tokens = name.split("_")
    if len(tokens) <= 8: return False
    for pat in KEEP_PATTERNS:
        if pat in tokens: return False
    return True
def normalise_name(name):
    if name in RENAME_EXACT: return RENAME_EXACT[name]
    for prefix, canonical in RENAME_PREFIX:
        if name.startswith(prefix): return canonical
    for pattern, canonical in RENAME_CONTAINS:
        if pattern in name: return canonical
    if is_paper_title(name): return "preamble"
    return name
def run_normalisation(workflow_id=DEFAULT_WORKFLOW_ID):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM Paper WHERE workflow_id=?", (workflow_id,))
    paper_ids = [r[0] for r in cur.fetchall()]
    print(f"[S4.5] Normalising {len(paper_ids)} papers (workflow {workflow_id})")
    deleted = renamed = merged = 0
    for paper_id in paper_ids:
        cur.execute("SELECT id, section_name FROM PaperContent WHERE paper_id=?", (paper_id,))
        for row_id, section_name in cur.fetchall():
            if is_noise(section_name):
                cur.execute("DELETE FROM PaperContent WHERE id=?", (row_id,))
                deleted += 1
        cur.execute("SELECT id, section_name FROM PaperContent WHERE paper_id=?", (paper_id,))
        for row_id, section_name in cur.fetchall():
            canonical = normalise_name(section_name)
            if canonical != section_name:
                cur.execute("UPDATE PaperContent SET section_name=? WHERE id=?", (canonical, row_id))
                renamed += 1
        cur.execute("SELECT section_name, COUNT(*) FROM PaperContent WHERE paper_id=? GROUP BY section_name HAVING COUNT(*) > 1", (paper_id,))
        for section_name, _ in cur.fetchall():
            cur.execute("SELECT id, content FROM PaperContent WHERE paper_id=? AND section_name=? ORDER BY id", (paper_id, section_name))
            dup_rows = cur.fetchall()
            keep_id = dup_rows[0][0]
            merged_content = " ".join(r[1] for r in dup_rows if r[1])
            cur.execute("UPDATE PaperContent SET content=? WHERE id=?", (merged_content[:50000], keep_id))
            for row_id, _ in dup_rows[1:]:
                cur.execute("DELETE FROM PaperContent WHERE id=?", (row_id,))
                merged += 1
    conn.commit()
    conn.close()
    print(f"[S4.5] Done - deleted={deleted} renamed={renamed} merged={merged}")
if __name__ == "__main__":
    wf = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKFLOW_ID
    run_normalisation(wf)
