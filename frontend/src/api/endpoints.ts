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
}

export const groundingApi = {
  run() {
    return api.post('/grounding/run')
  },
  results() {
    return api.get('/grounding/results')
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
}