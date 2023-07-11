import Head from 'next/head'
import styles from '../styles/Home.module.css'
import cx from 'classnames';
import { useEffect, useState } from 'react';
import { saveAs } from 'file-saver';
import { signIn, signOut, useSession } from 'next-auth/react'

export default function Index() {
  const { data: session } = useSession()
  if (session) {
    return <>
      Signed in as {session.user.email} <br />
      <button onClick={() => signOut()}>Sign out</button>
    </>
  }
  return <>
    Not signed in <br />
    <button onClick={() => signIn('google')}>Sign in</button>
  </>
}