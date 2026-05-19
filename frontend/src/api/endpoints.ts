import api from './index'

export const uploadApi = {
  upload(files: File[]) {
    const formData = new FormData()
    files.forEach((f) => formData.append('files', f))
    return api.post('/upload', formData)
  },
  list() {
    return api.get('/images')
  },
  reorder(order: number[]) {
    return api.post('/images/reorder', { order })
  },
  deleteImage(path: string) {
    return api.post('/images/delete', { path })
  },
  serveUrl(imgIdx: number) {
    return `/api/images/serve/${imgIdx}`
  },
  serveByNameUrl(filename: string) {
    return `/api/images/serve-by-name/${filename}`
  },
}

export const ocrApi = {
  run(onlyWhiteBg = false, zhTexts = true) {
    return api.post('/ocr/run', { only_white_bg: onlyWhiteBg, zh_texts: zhTexts })
  },
  results() {
    return api.get('/ocr/results')
  },
  updateText(imgIdx: number, textIdx: number, text: string) {
    return api.post('/ocr/update_text', { img_idx: imgIdx, text_idx: textIdx, text })
  },
  updateBox(imgIdx: number, boxIdx: number, box: number[]) {
    return api.post('/ocr/update_box', { img_idx: imgIdx, box_idx: boxIdx, box })
  },
  deleteBox(imgIdx: number, boxIdx: number) {
    return api.post('/ocr/delete_box', { img_idx: imgIdx, box_idx: boxIdx })
  },
  reorder(imgIdx: number, order: number[]) {
    return api.post('/ocr/reorder', { img_idx: imgIdx, order })
  },
  addBox(imgIdx: number, box: number[], text: string = '') {
    return api.post('/ocr/add_box', { img_idx: imgIdx, box, text })
  },
}

export const predictApi = {
  run() {
    return api.post('/predict/run')
  },
  results() {
    return api.get('/predict/results')
  },
  updateCharacterBox(imgIdx: number, charIdx: number, box: number[]) {
    return api.post('/predict/update_character_box', { img_idx: imgIdx, char_idx: charIdx, box })
  },
  deleteCharacter(imgIdx: number, charIdx: number) {
    return api.post('/predict/delete_character', { img_idx: imgIdx, char_idx: charIdx })
  },
  addCharacter(imgIdx: number, box: number[]) {
    return api.post('/predict/add_character', { img_idx: imgIdx, box })
  },
  updateAssociation(imgIdx: number, textIdx: number, charIdx: number | null) {
    return api.post('/predict/update_text_char_association', { img_idx: imgIdx, text_idx: textIdx, char_idx: charIdx })
  },
}

export const captionApi = {
  run(think = false) {
    return api.post('/caption/run', { think })
  },
  results() {
    return api.get('/caption/results')
  },
  updateCaption(imgIdx: number, caption: string) {
    return api.post('/caption/update_caption', { img_idx: imgIdx, caption })
  },
  updatePanelScript(imgIdx: number, panelScript: string) {
    return api.post('/caption/update_panel_script', { img_idx: imgIdx, panel_script: panelScript })
  },
}

export const groundingApi = {
  run(stylePrompt?: string) {
    return api.post('/grounding/run', { style_prompt: stylePrompt || null })
  },
  results() {
    return api.get('/grounding/results')
  },
  updateCaption(imgIdx: number, panelIdx: number, groundedCaption: string) {
    return api.post('/grounding/update_caption', { img_idx: imgIdx, panel_idx: panelIdx, grounded_caption: groundedCaption })
  },
  saveStylePrompt(stylePrompt: string) {
    return api.post('/grounding/save_style_prompt', { style_prompt: stylePrompt })
  },
}

export const proseApi = {
  buildScripts() {
    return api.post('/prose/build_scripts')
  },
  buildPrompt() {
    return api.post('/prose/build_prompt')
  },
  run() {
    return api.post('/prose/run')
  },
  results() {
    return api.get('/prose/results')
  },
  saveStoryBackground(storyBackground: string) {
    return api.post('/prose/save_story_background', { story_background: storyBackground })
  },
  saveProsePrompt(prosePromptText: string) {
    return api.post('/prose/save_prose_prompt', { prose_prompt_text: prosePromptText })
  },
  saveProseText(proseText: string) {
    return api.post('/prose/save_prose_text', { prose_text: proseText })
  },
  characterCrops(globalId: number) {
    return api.get(`/prose/character_crops/${globalId}`)
  },
  characterAllCrops(globalId: number) {
    return api.get(`/prose/character_all_crops/${globalId}`)
  },
  characterCropImage(globalId: number, cropKey: string) {
    return api.get(`/prose/character_crop_image/${globalId}/${cropKey}`)
  },
  generateReferences(globalId: number, views: string[], cropKeys?: string[], numPerView?: number, designImageFilenames?: string[]) {
    return api.post(`/prose/generate_references/${globalId}`, {
      views,
      crop_keys: cropKeys || [],
      num_per_view: numPerView || 1,
      design_image_filenames: designImageFilenames || [],
    })
  },
  referenceResults() {
    return api.get('/prose/reference_results')
  },
}

export const characterApi = {
  library() {
    return api.get('/character/library')
  },
  updateName(globalId: number, name: string) {
    return api.post('/character/update_name', { global_id: globalId, name })
  },
  updateGlobalId(imgIdx: number, charIdx: number, newGlobalId: number) {
    return api.post('/character/update_global_id', { img_idx: imgIdx, char_idx: charIdx, new_global_id: newGlobalId })
  },
  panelCharacters(imgIdx: number) {
    return api.get(`/character/panel_characters/${imgIdx}`)
  },
  addToLibrary() {
    return api.post('/character/library/add')
  },
  deleteFromLibrary(globalId: number) {
    return api.delete(`/character/library/${globalId}`)
  },
  designImages(globalId: number) {
    return api.get(`/character/${globalId}/design_images`)
  },
  uploadDesignImage(globalId: number, file: File) {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/character/${globalId}/design_upload`, formData)
  },
  deleteDesignImage(globalId: number, filename: string) {
    return api.delete(`/character/${globalId}/design_image/${filename}`)
  },
}

export const videoApi = {
  prepare() {
    return api.get('/video/prepare')
  },
  submit() {
    return api.post('/video/submit')
  },
  status() {
    return api.get('/video/status')
  },
  cancel() {
    return api.post('/video/cancel')
  },
  result() {
    return api.get('/video/result')
  },
}

export const projectApi = {
  list() {
    return api.get('/project/list')
  },
  create(name: string) {
    return api.post('/project/create', { name })
  },
  load(projectId: string) {
    return api.post(`/project/${projectId}/load`)
  },
  get(projectId: string) {
    return api.get(`/project/${projectId}`)
  },
  rename(projectId: string, name: string) {
    return api.post(`/project/${projectId}/rename`, { name })
  },
  delete(projectId: string) {
    return api.delete(`/project/${projectId}`)
  },
  current() {
    return api.get('/project/current')
  },
  exit() {
    return api.post('/project/exit')
  },
  applyMagiMode() {
    return api.post('/project/apply_magi_mode')
  },
}

export const configApi = {
  get() {
    return api.get('/config')
  },
  update(config: any) {
    return api.post('/config', config)
  },
  reset() {
    return api.post('/config/reset')
  },
  getMagiMode() {
    return api.get('/config/magi_v3_mode')
  },
}