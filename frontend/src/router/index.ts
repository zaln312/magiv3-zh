import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'upload',
      component: () => import('../views/UploadView.vue'),
    },
    {
      path: '/ocr',
      name: 'ocr',
      component: () => import('../views/OcrView.vue'),
    },
    {
      path: '/predict',
      name: 'predict',
      component: () => import('../views/PredictView.vue'),
    },
    {
      path: '/grounding',
      name: 'grounding',
      component: () => import('../views/GroundingView.vue'),
    },
    {
      path: '/prose',
      name: 'prose',
      component: () => import('../views/ProseView.vue'),
    },
  ],
})

export default router